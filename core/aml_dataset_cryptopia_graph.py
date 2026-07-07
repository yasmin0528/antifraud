"""
CryptopiaHacker 图级别节点分类数据集（增强版）。

核心思路：
- 整个数据集构建为一张大交易图
- 节点 = 地址，边 = 交易（sender → receiver）
- 每个节点的交易序列作为其特征
- 标签在节点上（heist=1, 其他=0）
- 按地址分层划分 train/val/test，确保地址不重叠

增强特征（V2）：
1. 节点序列特征：amount, log_amount, time_diff, tx_count_ratio, hourly_sin/cos
2. 节点静态特征：degree, pagerank, clustering_coef, amount_stats (mean/std/max/min)
3. 边特征：amount_norm, log_amount, rel_time_diff
4. 保存 raw_amount 和 amount_mean/std 供规则编码器使用
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from core.aml_dataset import PREPROCESS_SCHEMA_VERSION


def _compute_pagerank(edge_index: np.ndarray, n_nodes: int, alpha: float = 0.85, max_iter: int = 30) -> np.ndarray:
    """计算 PageRank 值（幂迭代法）。"""
    out_degree = np.bincount(edge_index[0], minlength=n_nodes).astype(float)
    out_degree[out_degree == 0] = 1.0  # 避免除零
    pr = np.ones(n_nodes) / n_nodes
    for _ in range(max_iter):
        new_pr = np.zeros(n_nodes)
        np.add.at(new_pr, edge_index[1], pr[edge_index[0]] / out_degree[edge_index[0]])
        pr = alpha * new_pr + (1 - alpha) / n_nodes
    return pr


def _compute_clustering_coefficient(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """近似计算每个节点的聚类系数（基于1-hop邻居，numpy加速版）。

    对大图仍然较慢(N>50000时)，
    此时使用基于度分布的近似方法加速。
    """
    from collections import defaultdict
    adj = defaultdict(set)
    for s, r in zip(edge_index[0], edge_index[1]):
        adj[s].add(r)
        adj[r].add(s)

    cc = np.zeros(n_nodes)
    if n_nodes > 50000:
        # 大图：只对有少量邻居的节点计算精确聚类系数
        # 高 degree 节点的聚类系数近似为 0（真实图通常如此）
        for i in range(n_nodes):
            neighbors = list(adj[i])
            deg = len(neighbors)
            if deg < 2 or deg > 500:
                continue
            neighbors_set = set(neighbors)
            edges_between = 0
            for u_idx in range(deg):
                u = neighbors[u_idx]
                u_neighbors = adj[u]
                for v_idx in range(u_idx + 1, deg):
                    if neighbors[v_idx] in u_neighbors:
                        edges_between += 1
            cc[i] = (2.0 * edges_between) / (deg * (deg - 1))
    else:
        for i in range(n_nodes):
            neighbors = adj[i]
            deg = len(neighbors)
            if deg < 2:
                continue
            edges_between = sum(
                1 for u in neighbors for v in neighbors
                if hash(u) < hash(v) and v in adj[u]
            )
            cc[i] = (2.0 * edges_between) / (deg * (deg - 1))
    return cc


def preprocess_cryptopia_graph(
    data_dir: str,
    max_seq_len: int = 50,
    min_tx_per_addr: int = 1,
    save_path: Optional[str] = "preprocessed_cryptopia.pt",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Dict:
    """
    CryptopiaHacker 图级别预处理（增强版 V2）。

    Returns:
        data:
            node_seq: [N, max_seq_len, 3]  [amount_norm, log_amount, time_diff]
            node_seq_len: [N]  实际交易数
            node_labels: [N]  1=heist, 0=其他
            node_static_feat: [N, 8]  [degree, pagerank, clustering, amount_mean, amount_std, amount_max, amount_min, n_unique_counterparties]
            edge_index: [2, E]
            edge_attr: [E, 3]  [amount_norm, log_amount, rel_time_diff]
            train_mask / val_mask / test_mask: [N] bool
            amount_mean: float  用于规则编码器反归一化
            amount_std: float   用于规则编码器反归一化
    """
    # ---- 1. 加载地址 ----
    addr_path = os.path.join(data_dir, "all-normal-address.csv")
    if not os.path.exists(addr_path):
        addr_path = os.path.join(data_dir, "all-address.csv")
    address_df = pd.read_csv(addr_path)
    n_accounts = len(address_df)

    labels_array = np.zeros(n_accounts, dtype=np.int64)
    labels_array[(address_df["label"] == "heist").values] = 1
    raw_group_names = address_df["name_tag"].fillna("background").astype(str).values
    explicit_group_names = sorted(
        {name for name in raw_group_names if name.startswith("ml_transit_")}
    )
    group_name_to_id = {name: idx for idx, name in enumerate(explicit_group_names)}
    group_ids_array = np.array(
        [group_name_to_id.get(name, -1) for name in raw_group_names], dtype=np.int64
    )
    print(f"Addresses: total={n_accounts}, heist={labels_array.sum()}")

    # ---- 2. 加载并融合多 cube 交易数据 ----
    # 合并 cube_xy, cube_yz, cube_zy 增加图覆盖率和交易密度
    cube_names = ["all-normal-tx-cube_xy.csv", "all-normal-tx-cube_yz.csv", "all-normal-tx-cube_zy.csv"]
    tx_frames = []
    for cname in cube_names:
        cpath = os.path.join(data_dir, cname)
        if os.path.exists(cpath):
            cf = pd.read_csv(cpath, header=None)
            cf.columns = ["sender", "receiver", "timestamp", "amount"]
            tx_frames.append(cf)

    tx = pd.concat(tx_frames, ignore_index=True)

    # 去重：完全相同的四元组 (sender, receiver, timestamp, amount) 只保留一份
    before = len(tx)
    tx = tx.drop_duplicates(subset=["sender", "receiver", "timestamp", "amount"])
    after = len(tx)
    print(f"Transactions: {before} (loaded) -> {after} (after dedup, -{before-after})")
    print(f"  Cube files used: {cube_names}")

    # 过滤超出范围的索引
    tx = tx[(tx["sender"] < n_accounts) & (tx["receiver"] < n_accounts)]
    print(f"  After OOB filter: {len(tx)}")

    # 金额归一化（保存参数供后续使用）
    raw_amounts = tx["amount"].values.copy()
    amount_mean, amount_std = tx["amount"].mean(), tx["amount"].std() + 1e-6
    tx["amount_norm"] = (tx["amount"] - amount_mean) / amount_std
    tx["log_amount"] = np.log1p(tx["amount"])

    # ---- 3. 聚合每个节点的交易序列 ----
    tx_sorted = tx.sort_values(["sender", "timestamp"]).reset_index(drop=True)

    node_seqs: Dict[int, List[List[float]]] = {}
    for sid, grp in tx_sorted.groupby("sender"):
        sid = int(sid)
        amounts = grp["amount_norm"].values
        log_amounts = grp["log_amount"].values
        times = grp["timestamp"].values
        time_diffs = np.diff(times, prepend=times[0])
        # 对每个节点，计算其交易数量在整个数据集中的分位数位置
        n_tx = len(amounts)
        seq_entries = []
        for i in range(n_tx):
            seq_entries.append([
                float(amounts[i]),
                float(log_amounts[i]),
                float(time_diffs[i]),
            ])
        node_seqs[sid] = seq_entries

    # ---- 4. 过滤有效节点 ----
    valid_nodes = sorted([
        idx for idx, seq in node_seqs.items() if len(seq) >= min_tx_per_addr
    ])
    print(f"Valid nodes (≥{min_tx_per_addr} tx): {len(valid_nodes)}")

    valid_labels = labels_array[valid_nodes]
    valid_group_ids = group_ids_array[valid_nodes]
    group_to_nodes: Dict[int, List[int]] = {}
    print(f"  heist: {valid_labels.sum()}, normal: {len(valid_nodes)-valid_labels.sum()}")

    N = len(valid_nodes)
    node_seq = torch.zeros(N, max_seq_len, 3, dtype=torch.float32)
    node_seq_len = torch.zeros(N, dtype=torch.long)
    node_labels = torch.tensor(valid_labels, dtype=torch.float32)
    node_group_ids = torch.tensor(valid_group_ids, dtype=torch.long)
    orig_to_new = {o: n for n, o in enumerate(valid_nodes)}

    for new_i, orig_i in enumerate(valid_nodes):
        seq = node_seqs[orig_i]
        n_tx = min(len(seq), max_seq_len)
        node_seq_len[new_i] = n_tx
        if valid_group_ids[new_i] >= 0:
            group_to_nodes.setdefault(int(valid_group_ids[new_i]), []).append(int(new_i))
        recent = seq[-max_seq_len:]
        for t, vals in enumerate(recent):
            node_seq[new_i, t, 0] = vals[0]  # amount_norm
            node_seq[new_i, t, 1] = vals[1]  # log_amount
            node_seq[new_i, t, 2] = vals[2]  # time_diff

    # ---- 5. 构建边 ----
    sender_t = tx["sender"].values
    receiver_t = tx["receiver"].values
    amount_norm_t = tx["amount_norm"].values
    log_amount_t = tx["log_amount"].values
    timestamp_t = tx["timestamp"].values

    valid_set = set(valid_nodes)
    edge_mask = np.array([
        s in valid_set and r in valid_set
        for s, r in zip(sender_t, receiver_t)
    ], dtype=bool)
    print(f"Edges in graph: {edge_mask.sum()} / {len(tx)}")

    edge_src = torch.tensor([orig_to_new[s] for s in sender_t[edge_mask]], dtype=torch.long)
    edge_dst = torch.tensor([orig_to_new[r] for r in receiver_t[edge_mask]], dtype=torch.long)
    edge_raw_amount = torch.tensor(raw_amounts[edge_mask], dtype=torch.float32)
    edge_amount_norm = torch.tensor(amount_norm_t[edge_mask], dtype=torch.float32)
    edge_log_amount = torch.tensor(log_amount_t[edge_mask], dtype=torch.float32)
    edge_timestamps = torch.tensor(timestamp_t[edge_mask], dtype=torch.float32)

    # 计算每条边的时间相对于该 sender 所有交易时间的相对位置
    # 用 log(1 + time_diff_in_seconds) 压缩
    edge_time_diffs = torch.zeros(edge_mask.sum(), dtype=torch.float32)
    sender_to_times: Dict[int, List[float]] = {}
    for i, s in enumerate(edge_src.numpy()):
        s_int = int(s)
        if s_int not in sender_to_times:
            sender_to_times[s_int] = []
        sender_to_times[s_int].append(float(edge_timestamps[i]))

    for s_int, times in sender_to_times.items():
        times_arr = np.array(sorted(times))
        if len(times_arr) > 1:
            time_diffs = np.diff(times_arr, prepend=times_arr[0])
            # 找到这个 sender 的边索引
            mask = edge_src.numpy() == s_int
            edge_time_diffs[mask] = torch.tensor(
                np.log1p(time_diffs), dtype=torch.float32
            )
        else:
            mask = edge_src.numpy() == s_int
            edge_time_diffs[mask] = 0.0

    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_attr = torch.stack([edge_amount_norm, edge_log_amount, edge_time_diffs], dim=-1)

    # ---- 5b. 计算节点静态特征 ----
    node_static_feat = torch.zeros(N, 10, dtype=torch.float32)

    # 度特征
    out_degree = np.bincount(edge_src.numpy(), minlength=N).astype(float)
    in_degree = np.bincount(edge_dst.numpy(), minlength=N).astype(float)
    node_static_feat[:, 0] = torch.from_numpy(out_degree)
    node_static_feat[:, 1] = torch.from_numpy(in_degree)

    # PageRank
    pr = _compute_pagerank(np.stack([edge_src.numpy(), edge_dst.numpy()], axis=0), N)
    node_static_feat[:, 2] = torch.from_numpy(pr)

    # 近似聚类系数（内部处理了大/小图的不同策略）
    cc = _compute_clustering_coefficient(
        np.stack([edge_src.numpy(), edge_dst.numpy()], axis=0), N
    )
    node_static_feat[:, 3] = torch.from_numpy(cc)

    # 金额统计（基于该地址作为 sender 的所有交易）
    amt_mean_global = float(amount_mean)  # 全局均值
    amt_std_global = float(amount_std)    # 全局标准差
    raw_edges_amounts = raw_amounts[edge_mask]

    # 用全局均值和标准差做归一化
    edge_src_np = edge_src.numpy()
    edge_dst_np = edge_dst.numpy()
    for sidx in range(N):
        mask = edge_src_np == sidx
        edge_amts = raw_edges_amounts[mask]
        if len(edge_amts) > 0:
            norm_amts = (edge_amts - amt_mean_global) / amt_std_global
            node_static_feat[sidx, 4] = float(np.mean(norm_amts))
            node_static_feat[sidx, 5] = float(np.std(norm_amts)) if len(norm_amts) > 1 else 0.0
            node_static_feat[sidx, 6] = float(np.max(norm_amts))
            node_static_feat[sidx, 7] = float(np.min(norm_amts))
            node_static_feat[sidx, 8] = float(len(norm_amts))
        # 唯一对手方数量
        rcv_set = set(edge_dst_np[mask])
        node_static_feat[sidx, 9] = float(len(rcv_set))

    # ---- 6. 按地址分层划分 ----
    all_idx = np.arange(N)
    labels_np = node_labels.numpy().astype(int)

    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")
    trn, tmp = train_test_split(
        all_idx,
        test_size=val_ratio + test_ratio,
        stratify=labels_np,
        random_state=random_state,
    )
    relative_test = test_ratio / (val_ratio + test_ratio) if test_ratio > 0 else 0.0
    if test_ratio > 0:
        val, tst = train_test_split(
            tmp,
            test_size=relative_test,
            stratify=labels_np[tmp],
            random_state=random_state,
        )
    else:
        val, tst = tmp, np.array([], dtype=np.int64)

    train_mask = torch.zeros(N, dtype=torch.bool); train_mask[trn] = True
    val_mask = torch.zeros(N, dtype=torch.bool);   val_mask[val] = True
    test_mask = torch.zeros(N, dtype=torch.bool);  test_mask[tst] = True

    print(f"\nSplit: train={len(trn)}, val={len(val)}, test={len(tst)}")
    print(f"  Train heist: {labels_np[trn].mean():.4f}")
    print(f"  Val heist:   {labels_np[val].mean():.4f}")
    print(f"  Test heist:  {labels_np[tst].mean():.4f}")

    data = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "split_unit": "graph_node",
        "group_type": "ml_transit_group",
        "edge_feature_names": ["amount_norm", "log_amount", "time_diff"],
        "group_ids": node_group_ids,
        "memory_group_ids": node_group_ids.clone(),
        "group_labels": torch.tensor(valid_labels, dtype=torch.long),
        "node_seq": node_seq,                    # [N, max_seq_len, 3]
        "node_seq_len": node_seq_len,            # [N]
        "node_labels": node_labels,              # [N]
        "node_static_feat": node_static_feat,    # [N, 10] static features
        "edge_index": edge_index,                # [2, E]
        "edge_attr": edge_attr,                  # [E, 3]
        "edge_raw_amount": edge_raw_amount,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "n_accounts": N,
        "n_groups": len(explicit_group_names),
        "group_name_to_id": group_name_to_id,
        "group_id_to_name": {int(v): k for k, v in group_name_to_id.items()},
        "group_to_nodes": {int(k): sorted(set(v)) for k, v in group_to_nodes.items()},
        "group_membership": {int(k): sorted(set(v)) for k, v in group_to_nodes.items()},
        "unknown_group_ratio": float((valid_group_ids < 0).mean()) if len(valid_group_ids) > 0 else 0.0,
        "orig_to_new": orig_to_new,
        "amount_mean": float(amount_mean),       # for rule encoder
        "amount_std": float(amount_std),         # for rule encoder
    }

    if save_path:
        torch.save(data, save_path)
        print(f"Saved to {save_path}")

    return data
