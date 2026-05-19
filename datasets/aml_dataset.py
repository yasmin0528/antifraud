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
    use_smote: bool = False,
    smote_ratio: float = 1.0,
    force_reprocess: bool = False,
) -> Dict:
    """
    完整数据预处理管线。

    1. 读取 CSV
    2. 排序、编码
    3. 特征工程（金额归一化、时间差）
    4. 可选 SMOTE 过采样
    5. 滑动窗口序列构建
    6. 保存为 .pt 文件

    Args:
        csv_path: 原始 CSV 路径
        window_size: 滑动窗口大小
        save_path: 保存路径（None 则不保存）
        use_smote: 是否使用 SMOTE
        smote_ratio: SMOTE 采样比例
        force_reprocess: 是否强制重新处理

    Returns:
        data: 包含以下 key 的字典
            - sequences: [N, window_size, 3]
            - labels: [N]
            - sender_idx: [N]
            - receiver_idx: [N]
            - alert_idx: [N]
            - edge_attr: [N, 2]
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

    # SMOTE
    if use_smote and _HAS_SMOTE:
        print(f"Applying SMOTE with ratio {smote_ratio}...")
        features = df[
            ["TX_AMOUNT_NORM", "TX_TYPE_ID", "TIME_DIFF", "SENDER_IDX", "RECEIVER_IDX", "ALERT_IDX"]
        ].values
        labels = df["IS_FRAUD"].values
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)

        df_resampled = pd.DataFrame(
            features_resampled,
            columns=["TX_AMOUNT_NORM", "TX_TYPE_ID", "TIME_DIFF", "SENDER_IDX", "RECEIVER_IDX", "ALERT_IDX"],
        )
        df_resampled["IS_FRAUD"] = labels_resampled
        df_resampled["TX_ID"] = np.arange(len(df_resampled))
        df_resampled["TIMESTAMP"] = np.arange(len(df_resampled))
        df = df_resampled
        print(f"After SMOTE: {len(df)} samples, fraud ratio: {df['IS_FRAUD'].mean():.3f}")
    elif use_smote and not _HAS_SMOTE:
        print("SMOTE requested but imblearn not available. Skipping SMOTE.")

    # 滑动窗口构建序列
    sequences = []
    sender_indices: list[int] = []
    receiver_indices: list[int] = []
    alert_indices: list[int] = []
    edge_attrs: list[list[float]] = []
    labels_list: list[int] = []

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

    data = {
        "sequences": torch.tensor(sequences, dtype=torch.float32),
        "labels": torch.tensor(labels_arr, dtype=torch.float32),
        "sender_idx": torch.tensor(sender_arr, dtype=torch.long),
        "receiver_idx": torch.tensor(receiver_arr, dtype=torch.long),
        "alert_idx": torch.tensor(alert_arr, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr_arr, dtype=torch.float32),
        "n_types": n_types_orig,
        "n_accounts": n_accounts,
        "n_groups": n_groups_orig,
    }

    if save_path:
        torch.save(data, save_path)
        print(f"Preprocessed data saved to {save_path}")

    return data


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
    unique_nodes = torch.unique(torch.cat([sender_idx, receiver_idx]))
    max_idx = int(unique_nodes.max().item())
    mapping = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)
    mapping[unique_nodes] = torch.arange(unique_nodes.size(0), device=device)

    sender_local = mapping[sender_idx]
    receiver_local = mapping[receiver_idx]
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

    train_dataset = TensorDataset(
        sequences, sender_idx, receiver_idx, edge_attr, alert_idx, labels
    )
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)

    train_sampler = get_sampler(labels[train_idx].numpy(), use_smote, smote_ratio)

    if train_sampler is not None:
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
        )
    else:
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
