"""
AML 数据集加载与预处理模块。

功能：
- 从 CSV 加载原始交易数据
- 特征工程（金额归一化、时间差、类型编码）
- 滑动窗口序列构建
- SMOTE 过采样
- 数据分割（train/val/test）
- Batch 图构建
- 交易摘要生成（用于 LLM 规则生成）
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from imblearn.over_sampling import SMOTE

    _HAS_SMOTE = True
except ImportError:
    SMOTE = None
    _HAS_SMOTE = False


def preprocess_data(
    csv_path: str,
    window_size: int = 10,
    save_path: Optional[str] = "preprocessed_data.pt",
    force_reprocess: bool = False,
) -> Dict:
    """
    完整数据预处理管线（不包含 SMOTE——SMOTE 仅对训练集执行）。

    流程：
    1. 读取 CSV
    2. 排序、编码
    3. 特征工程（金额归一化、时间差）
    4. 滑动窗口序列构建
    5. 保存为 .pt 文件

    Args:
        csv_path: 原始 CSV 路径
        window_size: 滑动窗口大小
        save_path: 保存路径（None 则不保存）
        force_reprocess: 是否强制重新处理

    Returns:
        data: 包含以下 key 的字典
            - sequences: [N, window_size, 3]
            - labels: [N]
            - sender_idx: [N]
            - receiver_idx: [N]
            - alert_idx: [N]
            - edge_attr: [N, 2]
            - detection_features: [N, 3]  原始特征（用于训练集SMOTE）
            - n_types, n_accounts, n_groups
    """
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    df = df.sort_values(["SENDER_ACCOUNT_ID", "TIMESTAMP"]).reset_index(drop=True)
    df["IS_FRAUD"] = df["IS_FRAUD"].astype(int)

    # 交易类型编码
    type_encoder = LabelEncoder()
    df["TX_TYPE_ID"] = type_encoder.fit_transform(df["TX_TYPE"])

    # 账户编码
    all_accounts = pd.concat(
        [df["SENDER_ACCOUNT_ID"], df["RECEIVER_ACCOUNT_ID"]], ignore_index=True
    )
    account_encoder = LabelEncoder()
    account_encoder.fit(all_accounts)
    df["SENDER_IDX"] = account_encoder.transform(df["SENDER_ACCOUNT_ID"])
    df["RECEIVER_IDX"] = account_encoder.transform(df["RECEIVER_ACCOUNT_ID"])

    # 特征工程
    df["TIME_DIFF"] = df.groupby("SENDER_ACCOUNT_ID")["TIMESTAMP"].diff().fillna(0)
    df["TX_AMOUNT_NORM"] = (df["TX_AMOUNT"] - df["TX_AMOUNT"].mean()) / (
        df["TX_AMOUNT"].std() + 1e-6
    )

    # Alert 编码
    alert_mask = df["ALERT_ID"] != -1
    df["ALERT_IDX"] = -1
    if alert_mask.any():
        alert_encoder = LabelEncoder()
        df.loc[alert_mask, "ALERT_IDX"] = alert_encoder.fit_transform(
            df.loc[alert_mask, "ALERT_ID"]
        )
    df["ALERT_IDX"] = df["ALERT_IDX"].fillna(-1).astype(int)

    n_types_orig = int(df["TX_TYPE_ID"].nunique())
    n_accounts = int(account_encoder.classes_.shape[0])
    n_groups_orig = int(df.loc[alert_mask, "ALERT_IDX"].nunique() if alert_mask.any() else 0)

    # 滑动窗口构建序列
    sequences = []
    sender_indices: list[int] = []
    receiver_indices: list[int] = []
    alert_indices: list[int] = []
    edge_attrs: list[list[float]] = []
    labels_list: list[int] = []
    # 存储滑动窗口最后一步的原始特征，用于训练集 SMOTE
    detection_features_list: list[list[float]] = []

    for _, group in df.groupby("SENDER_ACCOUNT_ID"):
        group = group.reset_index(drop=True)
        if len(group) < window_size:
            continue
        for i in range(len(group) - window_size + 1):
            window = group.iloc[i : i + window_size]
            features = window[["TX_AMOUNT_NORM", "TX_TYPE_ID", "TIME_DIFF"]].to_numpy(dtype=np.float32)
            sequences.append(features)
            labels_list.append(int(window["IS_FRAUD"].iat[-1]))
            sender_indices.append(int(window["SENDER_IDX"].iat[-1]))
            receiver_indices.append(int(window["RECEIVER_IDX"].iat[-1]))
            alert_indices.append(int(window["ALERT_IDX"].iat[-1]))
            edge_attrs.append(
                [float(window["TX_AMOUNT_NORM"].iat[-1]), float(window["TX_TYPE_ID"].iat[-1])]
            )
            # 存储最后一步的原始 3 维特征，用于 SMOTE
            last_row = window.iloc[-1]
            detection_features_list.append([
                float(last_row["TX_AMOUNT_NORM"]),
                float(last_row["TX_TYPE_ID"]),
                float(last_row["TIME_DIFF"]),
            ])

    if len(sequences) == 0:
        raise ValueError(
            "No valid sequences were created. Reduce window_size or check dataset size."
        )

    sequences = np.stack(sequences, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    sender_arr = np.array(sender_indices, dtype=np.int64)
    receiver_arr = np.array(receiver_indices, dtype=np.int64)
    alert_arr = np.array(alert_indices, dtype=np.int64)
    edge_attr_arr = np.stack(edge_attrs, axis=0)
    detection_features_arr = np.array(detection_features_list, dtype=np.float32)

    data = {
        "sequences": torch.tensor(sequences, dtype=torch.float32),
        "labels": torch.tensor(labels_arr, dtype=torch.float32),
        "sender_idx": torch.tensor(sender_arr, dtype=torch.long),
        "receiver_idx": torch.tensor(receiver_arr, dtype=torch.long),
        "alert_idx": torch.tensor(alert_arr, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr_arr, dtype=torch.float32),
        "detection_features": torch.tensor(detection_features_arr, dtype=torch.float32),
        "n_types": n_types_orig,
        "n_accounts": n_accounts,
        "n_groups": n_groups_orig,
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

    SMOTE 在滑动窗口后的特征空间生成合成样本，新样本的特征向量（序列）直接
    复制最后一步的特征填充整个窗口。

    Args:
        data: preprocess_data 的输出
        train_idx: 训练集索引
        smote_ratio: SMOTE 采样比例
        random_state: 随机种子

    Returns:
        data: 在训练集中插入了 SMOTE 合成样本的新数据字典
        train_idx: 扩展后的训练集索引（包含原始 + SMOTE 样本）
    """
    if not _HAS_SMOTE:
        print("SMOTE requested but imblearn not available. Skipping SMOTE.")
        return data, train_idx

    # 从 detection_features 中提取训练集样本的特征和标签用于 SMOTE
    detection_features = data["detection_features"].numpy()
    labels = data["labels"].numpy()

    train_features = detection_features[train_idx]
    train_labels = labels[train_idx]

    # 检查训练集是否有两类样本
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

    # 为合成样本构建序列：复制特征 feature_dim 次
    window_size = data["sequences"].size(1)
    synthetic_feats = features_resampled[n_original:]  # [n_synthetic, 3]

    syn_sequences = np.tile(
        synthetic_feats[:, np.newaxis, :], (1, window_size, 1)
    )  # [n_synthetic, window_size, 3]
    syn_labels = labels_resampled[n_original:]
    syn_sender = np.full(n_synthetic, -1, dtype=np.int64)
    syn_receiver = np.full(n_synthetic, -1, dtype=np.int64)
    syn_alert = np.full(n_synthetic, -1, dtype=np.int64)
    syn_edge_attr = synthetic_feats[:, :2]  # [n_synthetic, 2]  amount + type

    # 将合成样本拼接到原始数据末尾
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

    # 扩展训练集索引：原始训练集索引 + 新合成样本索引
    new_indices = np.arange(len(data["labels"]))
    synthetic_start = new_indices[-n_synthetic:]
    train_idx_ext = np.concatenate([train_idx, synthetic_start])

    fraud_ratio = data["labels"][train_idx_ext].mean().item()
    print(f"Train set after SMOTE: {len(train_idx_ext)} samples, fraud ratio={fraud_ratio:.3f}")

    return data, train_idx_ext


def build_batch_graph(
    sender_idx: torch.Tensor,
    receiver_idx: torch.Tensor,
    edge_attr: torch.Tensor,
    node_features: torch.Tensor,
    score_micro: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从批次数据构建子图。

    Args:
        sender_idx: [batch_size] 发送方索引
        receiver_idx: [batch_size] 接收方索引
        edge_attr: [batch_size, edge_dim] 边特征
        node_features: [batch_size, emb_dim] 节点特征
        score_micro: [batch_size, 1] CA1 微观异常分数

    Returns:
        node_x: [num_nodes, emb_dim + 1] 聚合后的节点特征
        edge_index: [2, num_edges] 重整后的边索引
        edge_attr_batch: [num_edges, edge_dim] 边特征
        sender_local: [batch_size] 发送方的本地节点索引
    """
    device = sender_idx.device
    batch_size = sender_idx.size(0)

    # 找出有效节点（sender_idx >= 0，排除 SMOTE 样本）
    valid_mask = sender_idx >= 0
    has_invalid = (~valid_mask).any()

    if has_invalid:
        # SMOTE 样本的 sender_idx=-1: 自成孤点，不参与图结构
        # 为它们分配新的唯一节点 ID
        max_orig = sender_idx.max().item()
        num_invalid = batch_size - valid_mask.sum().item()
        fake_ids = torch.arange(max_orig + 1, max_orig + 1 + num_invalid, device=device)
        sender_idx_fixed = sender_idx.clone()
        sender_idx_fixed[~valid_mask] = fake_ids[:num_invalid]
        receiver_idx_fixed = receiver_idx.clone()
        receiver_idx_fixed[~valid_mask] = fake_ids[:num_invalid]
    else:
        sender_idx_fixed = sender_idx
        receiver_idx_fixed = receiver_idx

    unique_nodes = torch.unique(torch.cat([sender_idx_fixed, receiver_idx_fixed]))
    max_idx = int(unique_nodes.max().item())
    mapping = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)
    mapping[unique_nodes] = torch.arange(unique_nodes.size(0), device=device)

    sender_local = mapping[sender_idx_fixed]
    receiver_local = mapping[receiver_idx_fixed]
    edge_index = torch.stack([sender_local, receiver_local], dim=0)

    combined = torch.cat([node_features, score_micro], dim=-1)
    num_nodes = unique_nodes.size(0)
    num_features = combined.size(-1)
    node_x = torch.zeros((num_nodes, num_features), device=device)
    counts = torch.zeros((num_nodes, 1), device=device)

    node_x.index_add_(0, sender_local, combined)
    counts.index_add_(
        0, sender_local, torch.ones((sender_local.size(0), 1), device=device)
    )
    node_x = node_x / counts.clamp(min=1.0)

    return node_x, edge_index, edge_attr, sender_local


def build_transaction_summary(
    edge_attr: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int = 1000,
) -> str:
    """
    从批次数据构建交易模式摘要（用于 LLM 规则生成）。

    Args:
        edge_attr: [batch_size, 2] 边特征（金额归一化值, 类型ID）
        labels: [batch_size] 标签
        max_samples: 最大采样数

    Returns:
        summary: 文本摘要
    """
    if edge_attr.numel() == 0:
        return "Normal transaction patterns observed."

    amounts = edge_attr[:, 0].cpu().numpy()

    if len(amounts) > max_samples:
        amounts = amounts[:max_samples]

    summary_lines = [
        f"- Total transactions analyzed: {len(amounts)}",
        f"- Amount range: ${amounts.min():.2f} ~ ${amounts.max():.2f}",
        f"- Average amount: ${amounts.mean():.2f}",
        f"- Median amount: ${float(np.median(amounts)):.2f}",
        f"- Transactions > $10,000: {(amounts > 10000).sum()} ({(amounts > 10000).mean() * 100:.1f}%)",
        f"- Transactions > $50,000: {(amounts > 50000).sum()} ({(amounts > 50000).mean() * 100:.1f}%)",
        f"- Transactions > $100,000: {(amounts > 100000).sum()} ({(amounts > 100000).mean() * 100:.1f}%)",
    ]

    if labels is not None:
        label_arr = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        if len(label_arr) > max_samples:
            label_arr = label_arr[:max_samples]
        fraud_count = label_arr.sum()
        summary_lines.append(
            f"- Fraudulent transactions: {fraud_count}/{len(label_arr)} "
            f"({fraud_count / len(label_arr) * 100:.1f}%)"
        )

    return "\n".join(summary_lines)


def make_data_splits(
    num_samples: int,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    分层数据分割。

    Returns:
        train_idx, val_idx, test_idx
    """
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


def get_sampler(
    labels: np.ndarray, use_smote: bool = False, smote_ratio: float = 1.0
) -> Optional[WeightedRandomSampler]:
    """
    获取加权采样器（SMOTE 替代方案）。

    Args:
        labels: 标签数组
        use_smote: 是否启用加权采样
        smote_ratio: 少数类权重系数

    Returns:
        WeightedRandomSampler 或 None
    """
    if not use_smote:
        return None

    labels = labels.astype(np.int64)
    class_sample_count = np.bincount(labels)
    if len(class_sample_count) < 2 or class_sample_count.min() == 0:
        sample_weights = np.ones_like(labels, dtype=np.float32)
    else:
        majority_label = int(np.argmax(class_sample_count))
        minority_label = 1 - majority_label
        ratio = max(0.01, min(float(smote_ratio), 10.0))
        weight = np.ones_like(labels, dtype=np.float32)
        weight[labels == minority_label] = (
            ratio * (class_sample_count[majority_label] / class_sample_count[minority_label])
        )
        sample_weights = weight

    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def get_dataloaders(
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

    Args:
        data: preprocess_data 的输出
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        batch_size: 批次大小
        use_smote: 是否使用加权采样
        smote_ratio: SMOTE 比例
        random_state: 随机种子
        num_workers: DataLoader workers

    Returns:
        train_loader, val_loader, test_loader, metadata
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

    # ---- SMOTE 仅应用在训练集上 ----
    if use_smote:
        data, train_idx = apply_smote_to_train(
            data, train_idx, smote_ratio=smote_ratio, random_state=random_state,
        )
        # 使用扩展后的数据重新取 tensor
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
