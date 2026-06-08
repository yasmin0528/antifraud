"""
CryptopiaHacker 数据集加载与预处理模块。

该数据集与原始 AML 数据集不同：
- 标签在地址级别（地址是 hacker/heist 还是 benign），而非交易级别
- 交易数据以 3 个 cube 文件（_xy, _yz, _zy）的形式给出，是边表列表
- cube 文件中的 sender/receiver 索引 == all-normal-address.csv 的行号

适配方案：
1. 使用 cube_xy.csv 的 [sender, receiver, timestamp, amount] 构建交易序列
2. 按 sender 分组，timestamp 排序，滑动窗口构造序列
3. 标签来自 all-normal-address.csv 的 label 列（heist=1, else=0）
4. 每条序列的标签 = 该序列最后一个交易的 sender 地址的标签
5. 缺失 TX_TYPE，用常量 0 替代；缺失 ALERT_ID，用 -1 替代
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

try:
    from imblearn.over_sampling import SMOTE

    _HAS_SMOTE = True
except ImportError:
    SMOTE = None
    _HAS_SMOTE = False


def preprocess_cryptopia_data(
    data_dir: str,
    window_size: int = 5,
    save_path: Optional[str] = "preprocessed_cryptopia.pt",
    force_reprocess: bool = False,
) -> Dict:
    """
    CryptopiaHacker 数据预处理管线。

    流程：
    1. 加载 all-normal-address.csv 获取地址标签
    2. 加载 cube_xy.csv 获取交易边表
    3. 按 sender 分组 + timestamp 排序 + 滑动窗口构建序列
    4. 地址标签映射到序列（序列最后一个交易的 sender 地址决定标签）

    Args:
        data_dir: CryptopiaHacker 数据目录
        window_size: 滑动窗口大小
        save_path: 保存路径（None 则不保存）
        force_reprocess: 是否强制重新处理

    Returns:
        data: 包含以下 key 的字典
            - sequences: [N, window_size, 3]  [amount_norm, tx_type=0, time_diff]
            - labels: [N]  地址级标签 (1=heist, 0=其他)
            - sender_idx: [N]  地址索引（在 all-normal-address 中的行号）
            - receiver_idx: [N]  接收方地址索引
            - alert_idx: [N]  全 -1（无 alert 信息）
            - edge_attr: [N, 2]  [amount_norm, tx_type=0]
            - detection_features: [N, 3]  最后一步的原始特征
            - n_types: 0（无交易类型信息）
            - n_accounts: int  唯一地址数
            - n_groups: 0（无 alert 分组）
    """
    # ---- 1. 加载地址标签 ----
    addr_path = os.path.join(data_dir, "all-normal-address.csv")
    if not os.path.exists(addr_path):
        # fallback: 尝试 all-address.csv
        addr_path = os.path.join(data_dir, "all-address.csv")

    address_df = pd.read_csv(addr_path)
    n_accounts = len(address_df)

    # 标签：heist=1（正类），其余=0（负类）
    labels_array = np.zeros(n_accounts, dtype=np.int64)
    heist_mask = address_df["label"] == "heist"
    labels_array[heist_mask.values] = 1

    n_heist = heist_mask.sum()
    n_normal = n_accounts - n_heist
    print(f"Addresses: total={n_accounts}, heist={n_heist}, normal={n_normal}")

    # ---- 2. 加载交易边表 ----
    cube_path = os.path.join(data_dir, "all-normal-tx-cube_xy.csv")
    if not os.path.exists(cube_path):
        raise FileNotFoundError(f"Cube file not found: {cube_path}")

    # cube_xy: [sender_idx, receiver_idx, timestamp, amount] 无表头
    tx = pd.read_csv(cube_path, header=None)
    tx.columns = ["sender", "receiver", "timestamp", "amount"]

    print(f"Transactions loaded: {len(tx)}")
    print(f"Sender range: [{tx['sender'].min()}, {tx['sender'].max()}]")
    print(f"Receiver range: [{tx['receiver'].min()}, {tx['receiver'].max()}]")

    # ---- 3. 特征工程 ----
    # 按 sender + timestamp 排序
    tx = tx.sort_values(["sender", "timestamp"]).reset_index(drop=True)

    # 金额归一化（全局 z-score）
    amount_mean = tx["amount"].mean()
    amount_std = tx["amount"].std() + 1e-6
    tx["amount_norm"] = (tx["amount"] - amount_mean) / amount_std

    # 时间差（同 sender 内，与前一笔交易的时间间隔）
    tx["time_diff"] = tx.groupby("sender")["timestamp"].diff().fillna(0)

    # 没有交易类型，用 0 填充
    tx["tx_type"] = 0

    print(f"Amount: mean={amount_mean:.4f}, std={amount_std:.4f}")
    print(f"Time diff: max={tx['time_diff'].max():.0f}, median={tx['time_diff'].median():.0f}")

    # ---- 4. 滑动窗口构建序列 ----
    sequences = []
    sender_indices: list[int] = []
    receiver_indices: list[int] = []
    alert_indices: list[int] = []
    edge_attrs: list[list[float]] = []
    labels_list: list[int] = []
    detection_features_list: list[list[float]] = []

    for sender_id, group in tx.groupby("sender"):
        group = group.reset_index(drop=True)
        if len(group) < window_size:
            continue

        # 该 sender 的地址级标签
        if sender_id < len(labels_array):
            addr_label = int(labels_array[sender_id])
        else:
            addr_label = 0  # 默认正常

        for i in range(len(group) - window_size + 1):
            window = group.iloc[i : i + window_size]
            features = window[["amount_norm", "tx_type", "time_diff"]].to_numpy(dtype=np.float32)
            sequences.append(features)
            labels_list.append(addr_label)
            sender_indices.append(int(window["sender"].iat[-1]))
            receiver_indices.append(int(window["receiver"].iat[-1]))
            alert_indices.append(-1)  # 无 alert 信息
            edge_attrs.append([
                float(window["amount_norm"].iat[-1]),
                float(window["tx_type"].iat[-1]),
            ])
            last_row = window.iloc[-1]
            detection_features_list.append([
                float(last_row["amount_norm"]),
                float(last_row["tx_type"]),
                float(last_row["time_diff"]),
            ])

    if len(sequences) == 0:
        raise ValueError(
            f"No valid sequences created. window_size={window_size} may be too large "
            f"or dataset has no senders with enough transactions."
        )

    sequences_arr = np.stack(sequences, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    sender_arr = np.array(sender_indices, dtype=np.int64)
    receiver_arr = np.array(receiver_indices, dtype=np.int64)
    alert_arr = np.array(alert_indices, dtype=np.int64)
    edge_attr_arr = np.stack(edge_attrs, axis=0)
    detection_arr = np.array(detection_features_list, dtype=np.float32)

    # ---- 统计 ----
    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos
    n_unique_senders = len(np.unique(sender_arr))

    print(f"\nSequences built: total={len(sequences_arr)}, window_size={window_size}")
    print(f"Positive (heist): {n_pos} ({100*n_pos/len(sequences_arr):.1f}%)")
    print(f"Negative: {n_neg} ({100*n_neg/len(sequences_arr):.1f}%)")
    print(f"Unique senders in sequences: {n_unique_senders}")
    print(f"Sequence feature dim: {sequences_arr.shape[2]}")

    data = {
        "sequences": torch.tensor(sequences_arr, dtype=torch.float32),
        "labels": torch.tensor(labels_arr, dtype=torch.float32),
        "sender_idx": torch.tensor(sender_arr, dtype=torch.long),
        "receiver_idx": torch.tensor(receiver_arr, dtype=torch.long),
        "alert_idx": torch.tensor(alert_arr, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr_arr, dtype=torch.float32),
        "detection_features": torch.tensor(detection_arr, dtype=torch.float32),
        "n_types": 0,        # 无交易类型信息
        "n_accounts": n_accounts,
        "n_groups": 0,       # 无 alert 分组
    }

    if save_path:
        torch.save(data, save_path)
        print(f"Preprocessed data saved to {save_path}")

    return data


def apply_smote_to_train(
    data: Dict,
    train_idx: np.ndarray,
    smote_ratio: float = 1.0,
    random_state: int = 42,
) -> Tuple[Dict, np.ndarray]:
    """
    仅对训练集应用 SMOTE 过采样。

    与 aml_dataset.py 中相同，但 tx_type 始终为 0。
    """
    if not _HAS_SMOTE:
        print("SMOTE requested but imblearn not available. Skipping SMOTE.")
        return data, train_idx

    detection_features = data["detection_features"].numpy()
    labels = data["labels"].numpy()
    sequences = data["sequences"].numpy()

    train_features = detection_features[train_idx]
    train_labels = labels[train_idx]
    train_sequences = sequences[train_idx]

    unique_classes = np.unique(train_labels)
    if len(unique_classes) < 2:
        print(f"Warning: train set has only {len(unique_classes)} class, skipping SMOTE.")
        return data, train_idx

    print(f"Applying SMOTE on train set (ratio={smote_ratio})...")
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    features_resampled, labels_resampled = smote.fit_resample(train_features, train_labels)

    n_original = len(train_idx)
    n_synthetic = len(features_resampled) - n_original

    if n_synthetic <= 0:
        print("No synthetic samples generated by SMOTE.")
        return data, train_idx

    print(f"SMOTE generated {n_synthetic} synthetic samples.")

    window_size = data["sequences"].size(1)
    synthetic_feats = features_resampled[n_original:]

    rng = np.random.RandomState(random_state)
    donor_indices = rng.randint(0, len(train_idx), size=n_synthetic)
    donor_sequences = train_sequences[donor_indices]

    syn_sequences = donor_sequences.copy()
    syn_sequences[:, -1, :] = synthetic_feats
    syn_labels = labels_resampled[n_original:]
    syn_sender = np.full(n_synthetic, -1, dtype=np.int64)
    syn_receiver = np.full(n_synthetic, -1, dtype=np.int64)
    syn_alert = np.full(n_synthetic, -1, dtype=np.int64)
    syn_edge_attr = synthetic_feats[:, :2]

    data["sequences"] = torch.cat([
        data["sequences"],
        torch.tensor(syn_sequences, dtype=torch.float32),
    ], dim=0)
    data["labels"] = torch.cat([
        data["labels"],
        torch.tensor(syn_labels, dtype=torch.float32),
    ], dim=0)
    data["sender_idx"] = torch.cat([
        data["sender_idx"],
        torch.tensor(syn_sender, dtype=torch.long),
    ], dim=0)
    data["receiver_idx"] = torch.cat([
        data["receiver_idx"],
        torch.tensor(syn_receiver, dtype=torch.long),
    ], dim=0)
    data["alert_idx"] = torch.cat([
        data["alert_idx"],
        torch.tensor(syn_alert, dtype=torch.long),
    ], dim=0)
    data["edge_attr"] = torch.cat([
        data["edge_attr"],
        torch.tensor(syn_edge_attr, dtype=torch.float32),
    ], dim=0)
    data["detection_features"] = torch.cat([
        data["detection_features"],
        torch.tensor(synthetic_feats, dtype=torch.float32),
    ], dim=0)

    new_indices = np.arange(len(data["labels"]))
    synthetic_start = new_indices[-n_synthetic:]
    train_idx_ext = np.concatenate([train_idx, synthetic_start])

    fraud_ratio = data["labels"][train_idx_ext].mean().item()
    print(f"Train set after SMOTE: {len(train_idx_ext)} samples, fraud ratio={fraud_ratio:.3f}")

    return data, train_idx_ext


def make_data_splits(
    num_samples: int,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """分层数据分割（同原始实现）。"""
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    train_idx, temp_idx = train_test_split(
        np.arange(num_samples),
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=random_state,
    )

    if test_ratio > 0:
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            stratify=labels[temp_idx],
            random_state=random_state,
        )
    else:
        val_idx, test_idx = temp_idx, np.array([], dtype=np.int64)

    return train_idx, val_idx, test_idx


def get_cryptopia_dataloaders(
    data: Dict,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 32,
    use_smote: bool = False,
    smote_ratio: float = 1.0,
    random_state: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict]:
    """
    从预处理数据创建 DataLoader。

    与 get_dataloaders 接口完全相同。
    """
    sequences = data["sequences"]
    labels = data["labels"]
    sender_idx = data["sender_idx"]
    receiver_idx = data["receiver_idx"]
    alert_idx = data["alert_idx"]
    edge_attr = data["edge_attr"]

    num_samples = labels.size(0)
    train_idx, val_idx, test_idx = make_data_splits(
        num_samples,
        labels.numpy(),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    if use_smote:
        data, train_idx = apply_smote_to_train(
            data, train_idx, smote_ratio=smote_ratio, random_state=random_state,
        )
        sequences = data["sequences"]
        labels = data["labels"]
        sender_idx = data["sender_idx"]
        receiver_idx = data["receiver_idx"]
        alert_idx = data["alert_idx"]
        edge_attr = data["edge_attr"]

    train_dataset = TensorDataset(
        sequences, sender_idx, receiver_idx, edge_attr, alert_idx, labels
    )
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = None
    if len(val_idx) > 0:
        val_dataset = torch.utils.data.Subset(train_dataset, val_idx)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    test_loader = None
    if len(test_idx) > 0:
        test_dataset = torch.utils.data.Subset(train_dataset, test_idx)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    metadata = {
        "n_types": data.get("n_types", 0),
        "n_accounts": data.get("n_accounts", 0),
        "n_groups": data.get("n_groups", 0),
        "num_samples": num_samples,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }

    return train_loader, val_loader, test_loader, metadata
