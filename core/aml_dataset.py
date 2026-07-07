"""
AML dataset preprocessing and dataloader helpers.

The tabular AML pipeline now uses account-level graph semantics:
- samples remain transaction windows for supervision
- graph nodes are accounts
- splits are performed by sender account to avoid window leakage
"""

from __future__ import annotations

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


PREPROCESS_SCHEMA_VERSION = 3


def preprocess_data(
    csv_path: str,
    window_size: int = 10,
    save_path: Optional[str] = "preprocessed_data.pt",
    force_reprocess: bool = False,
) -> Dict:
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    df = df.sort_values(["SENDER_ACCOUNT_ID", "TIMESTAMP"]).reset_index(drop=True)
    df["IS_FRAUD"] = df["IS_FRAUD"].astype(int)

    type_encoder = LabelEncoder()
    df["TX_TYPE_ID"] = type_encoder.fit_transform(df["TX_TYPE"])

    all_accounts = pd.concat(
        [df["SENDER_ACCOUNT_ID"], df["RECEIVER_ACCOUNT_ID"]], ignore_index=True
    )
    account_encoder = LabelEncoder()
    account_encoder.fit(all_accounts)
    df["SENDER_IDX"] = account_encoder.transform(df["SENDER_ACCOUNT_ID"])
    df["RECEIVER_IDX"] = account_encoder.transform(df["RECEIVER_ACCOUNT_ID"])

    df["TIME_DIFF"] = df.groupby("SENDER_ACCOUNT_ID")["TIMESTAMP"].diff().fillna(0)
    amount_mean = float(df["TX_AMOUNT"].mean())
    amount_std = float(df["TX_AMOUNT"].std() + 1e-6)
    df["TX_AMOUNT_NORM"] = (df["TX_AMOUNT"] - amount_mean) / amount_std
    df["TX_LOG_AMOUNT"] = np.log1p(df["TX_AMOUNT"].clip(lower=0.0))

    alert_mask = df["ALERT_ID"] != -1
    df["ALERT_IDX"] = -1
    if alert_mask.any():
        alert_encoder = LabelEncoder()
        df.loc[alert_mask, "ALERT_IDX"] = alert_encoder.fit_transform(
            df.loc[alert_mask, "ALERT_ID"]
        )
    df["ALERT_IDX"] = df["ALERT_IDX"].fillna(-1).astype(int)

    n_types = int(df["TX_TYPE_ID"].nunique())
    n_accounts = int(account_encoder.classes_.shape[0])
    n_groups = int(df.loc[alert_mask, "ALERT_IDX"].nunique() if alert_mask.any() else 0)

    sequences = []
    sender_indices: list[int] = []
    receiver_indices: list[int] = []
    alert_indices: list[int] = []
    edge_attrs: list[list[float]] = []
    labels_list: list[int] = []
    detection_features_list: list[list[float]] = []
    edge_raw_amounts: list[float] = []

    account_sequences = np.zeros((n_accounts, window_size, 3), dtype=np.float32)
    account_seq_len = np.zeros(n_accounts, dtype=np.int64)
    account_alert_idx = np.full(n_accounts, -1, dtype=np.int64)
    alert_to_accounts: Dict[int, list[int]] = {}
    alert_to_samples: Dict[int, list[int]] = {}

    for _, group in df.groupby("SENDER_ACCOUNT_ID"):
        group = group.reset_index(drop=True)
        sender_idx = int(group["SENDER_IDX"].iat[0])

        sender_seq = group[["TX_AMOUNT_NORM", "TX_TYPE_ID", "TIME_DIFF"]].to_numpy(
            dtype=np.float32
        )
        sender_recent = sender_seq[-window_size:]
        account_seq_len[sender_idx] = len(sender_recent)
        account_sequences[sender_idx, : len(sender_recent)] = sender_recent
        valid_alerts = group.loc[group["ALERT_IDX"] >= 0, "ALERT_IDX"].astype(int).tolist()
        if valid_alerts:
            account_alert_idx[sender_idx] = int(valid_alerts[-1])
            for alert_idx in sorted(set(valid_alerts)):
                alert_to_accounts.setdefault(int(alert_idx), []).append(sender_idx)

        if len(group) < window_size:
            continue

        for i in range(len(group) - window_size + 1):
            window = group.iloc[i : i + window_size]
            features = window[["TX_AMOUNT_NORM", "TX_TYPE_ID", "TIME_DIFF"]].to_numpy(
                dtype=np.float32
            )
            sequences.append(features)
            labels_list.append(int(window["IS_FRAUD"].iat[-1]))
            sender_indices.append(int(window["SENDER_IDX"].iat[-1]))
            receiver_indices.append(int(window["RECEIVER_IDX"].iat[-1]))
            alert_indices.append(int(window["ALERT_IDX"].iat[-1]))
            sample_idx = len(labels_list) - 1
            if alert_indices[-1] >= 0:
                alert_to_samples.setdefault(int(alert_indices[-1]), []).append(sample_idx)
            edge_attrs.append(
                [
                    float(window["TX_AMOUNT_NORM"].iat[-1]),
                    float(window["TX_LOG_AMOUNT"].iat[-1]),
                    float(window["TIME_DIFF"].iat[-1]),
                ]
            )
            edge_raw_amounts.append(float(window["TX_AMOUNT"].iat[-1]))
            last_row = window.iloc[-1]
            detection_features_list.append(
                [
                    float(last_row["TX_AMOUNT_NORM"]),
                    float(last_row["TX_TYPE_ID"]),
                    float(last_row["TIME_DIFF"]),
                ]
            )

    if not sequences:
        raise ValueError("No valid AML sequences created. Reduce window_size or inspect the dataset.")

    data = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "split_unit": "sender_account",
        "group_type": "alert_id",
        "edge_feature_names": ["amount_norm", "log_amount", "time_diff"],
        "sample_group_ids": torch.tensor(np.array(sender_indices, dtype=np.int64), dtype=torch.long),
        "split_group_ids": torch.tensor(np.array(sender_indices, dtype=np.int64), dtype=torch.long),
        "group_ids": torch.tensor(np.array(alert_indices, dtype=np.int64), dtype=torch.long),
        "memory_group_ids": torch.tensor(np.array(alert_indices, dtype=np.int64), dtype=torch.long),
        "group_labels": torch.arange(max(n_groups, 0), dtype=torch.long),
        "sequences": torch.tensor(np.stack(sequences, axis=0), dtype=torch.float32),
        "labels": torch.tensor(np.array(labels_list, dtype=np.int64), dtype=torch.float32),
        "sender_idx": torch.tensor(np.array(sender_indices, dtype=np.int64), dtype=torch.long),
        "receiver_idx": torch.tensor(np.array(receiver_indices, dtype=np.int64), dtype=torch.long),
        "alert_idx": torch.tensor(np.array(alert_indices, dtype=np.int64), dtype=torch.long),
        "edge_attr": torch.tensor(np.stack(edge_attrs, axis=0), dtype=torch.float32),
        "edge_raw_amount": torch.tensor(np.array(edge_raw_amounts, dtype=np.float32), dtype=torch.float32),
        "detection_features": torch.tensor(np.array(detection_features_list, dtype=np.float32), dtype=torch.float32),
        "account_sequences": torch.tensor(account_sequences, dtype=torch.float32),
        "account_seq_len": torch.tensor(account_seq_len, dtype=torch.long),
        "account_alert_idx": torch.tensor(account_alert_idx, dtype=torch.long),
        "alert_to_accounts": {int(k): sorted(set(v)) for k, v in alert_to_accounts.items()},
        "alert_to_samples": {int(k): sorted(set(v)) for k, v in alert_to_samples.items()},
        "group_membership": {int(k): sorted(set(v)) for k, v in alert_to_samples.items()},
        "amount_mean": amount_mean,
        "amount_std": amount_std,
        "n_types": n_types,
        "n_accounts": n_accounts,
        "n_groups": n_groups,
        "smote_applied": False,
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
    if not _HAS_SMOTE:
        print("SMOTE requested but imblearn not available. Skipping SMOTE.")
        return data, train_idx

    detection_features = data["detection_features"].numpy()
    labels = data["labels"].numpy()
    sequences = data["sequences"].numpy()

    train_features = detection_features[train_idx]
    train_labels = labels[train_idx]
    train_sequences = sequences[train_idx]

    if len(np.unique(train_labels)) < 2:
        print("Warning: train set has only one class, skipping SMOTE.")
        return data, train_idx

    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    features_resampled, labels_resampled = smote.fit_resample(train_features, train_labels)

    n_original = len(train_idx)
    n_synthetic = len(features_resampled) - n_original
    if n_synthetic <= 0:
        return data, train_idx

    rng = np.random.RandomState(random_state)
    donor_indices = rng.randint(0, len(train_idx), size=n_synthetic)
    donor_sequences = train_sequences[donor_indices]

    syn_sequences = donor_sequences.copy()
    syn_sequences[:, -1, :] = features_resampled[n_original:]
    syn_labels = labels_resampled[n_original:]
    syn_sender = np.full(n_synthetic, -1, dtype=np.int64)
    syn_receiver = np.full(n_synthetic, -1, dtype=np.int64)
    syn_alert = np.full(n_synthetic, -1, dtype=np.int64)
    syn_edge_attr = np.stack(
        [
            features_resampled[n_original:, 0],
            np.zeros(n_synthetic, dtype=np.float32),
            features_resampled[n_original:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    syn_raw_amount = np.zeros(n_synthetic, dtype=np.float32)
    syn_split_group_ids = np.full(n_synthetic, -1, dtype=np.int64)
    syn_group_ids = np.full(n_synthetic, -1, dtype=np.int64)

    data["sequences"] = torch.cat(
        [data["sequences"], torch.tensor(syn_sequences, dtype=torch.float32)], dim=0
    )
    data["labels"] = torch.cat(
        [data["labels"], torch.tensor(syn_labels, dtype=torch.float32)], dim=0
    )
    data["sender_idx"] = torch.cat(
        [data["sender_idx"], torch.tensor(syn_sender, dtype=torch.long)], dim=0
    )
    data["receiver_idx"] = torch.cat(
        [data["receiver_idx"], torch.tensor(syn_receiver, dtype=torch.long)], dim=0
    )
    data["alert_idx"] = torch.cat(
        [data["alert_idx"], torch.tensor(syn_alert, dtype=torch.long)], dim=0
    )
    data["edge_attr"] = torch.cat(
        [data["edge_attr"], torch.tensor(syn_edge_attr, dtype=torch.float32)], dim=0
    )
    data["edge_raw_amount"] = torch.cat(
        [data["edge_raw_amount"], torch.tensor(syn_raw_amount, dtype=torch.float32)], dim=0
    )
    data["detection_features"] = torch.cat(
        [data["detection_features"], torch.tensor(features_resampled[n_original:], dtype=torch.float32)], dim=0
    )
    data["sample_group_ids"] = torch.cat(
        [data["sample_group_ids"], torch.tensor(syn_split_group_ids, dtype=torch.long)], dim=0
    )
    if "split_group_ids" in data:
        data["split_group_ids"] = torch.cat(
            [data["split_group_ids"], torch.tensor(syn_split_group_ids, dtype=torch.long)], dim=0
        )
    if "group_ids" in data:
        data["group_ids"] = torch.cat(
            [data["group_ids"], torch.tensor(syn_group_ids, dtype=torch.long)], dim=0
        )
    data["smote_applied"] = True
    data["original_train_size"] = int(n_original)
    data["resampled_train_size"] = int(len(train_idx) + n_synthetic)

    synthetic_start = np.arange(len(data["labels"]))[-n_synthetic:]
    train_idx_ext = np.concatenate([train_idx, synthetic_start])
    return data, train_idx_ext


def build_batch_graph(
    sender_idx: torch.Tensor,
    receiver_idx: torch.Tensor,
    edge_attr: torch.Tensor,
    account_ids: torch.Tensor,
    account_features: torch.Tensor,
    account_scores: torch.Tensor,
    fallback_features: Optional[torch.Tensor] = None,
    fallback_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sender_idx.device
    feature_dim = account_features.size(-1)
    score_dim = account_scores.size(-1)

    unique_valid = account_ids
    base_nodes = unique_valid.tolist()
    next_fake_id = (max(base_nodes) + 1) if base_nodes else 0

    sender_fixed = sender_idx.clone()
    receiver_fixed = receiver_idx.clone()
    fake_node_positions = []

    for idx in range(sender_idx.size(0)):
        if sender_fixed[idx] < 0 or receiver_fixed[idx] < 0:
            fake_id = next_fake_id
            next_fake_id += 1
            sender_fixed[idx] = fake_id
            receiver_fixed[idx] = fake_id
            fake_node_positions.append(idx)

    unique_nodes = torch.unique(torch.cat([unique_valid, sender_fixed, receiver_fixed]))
    max_idx = int(unique_nodes.max().item()) if unique_nodes.numel() > 0 else -1
    mapping = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)
    mapping[unique_nodes] = torch.arange(unique_nodes.size(0), device=device)

    sender_local = mapping[sender_fixed]
    receiver_local = mapping[receiver_fixed]
    edge_index = torch.stack([sender_local, receiver_local], dim=0)

    node_x = torch.zeros((unique_nodes.size(0), feature_dim + score_dim), device=device)
    if unique_valid.numel() > 0:
        local_valid = mapping[unique_valid]
        node_x[local_valid] = torch.cat([account_features, account_scores], dim=-1)

    if fake_node_positions and fallback_features is not None and fallback_scores is not None:
        for sample_idx in fake_node_positions:
            local_idx = sender_local[sample_idx]
            node_x[local_idx] = torch.cat(
                [fallback_features[sample_idx], fallback_scores[sample_idx]], dim=-1
            )

    return node_x, edge_index, edge_attr, sender_local


def build_transaction_summary(
    edge_attr: torch.Tensor,
    labels: Optional[torch.Tensor],
    raw_amounts: Optional[torch.Tensor] = None,
    max_samples: int = 1000,
) -> str:
    if edge_attr.numel() == 0:
        return "Normal transaction patterns observed."

    if raw_amounts is not None:
        amounts = raw_amounts.detach().cpu().numpy()
    elif edge_attr.size(-1) > 1:
        amounts = torch.expm1(edge_attr[:, 1].detach().cpu()).numpy()
    else:
        amounts = edge_attr[:, 0].detach().cpu().numpy()

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
        label_arr = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        label_arr = label_arr[:max_samples]
        fraud_count = label_arr.sum()
        summary_lines.append(
            f"- Fraudulent transactions: {fraud_count}/{len(label_arr)} ({fraud_count / len(label_arr) * 100:.1f}%)"
        )
    return "\n".join(summary_lines)


def make_group_splits(
    group_ids: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    valid_mask = group_ids >= 0
    valid_groups = np.unique(group_ids[valid_mask])
    group_labels = np.array([labels[group_ids == gid].max() for gid in valid_groups], dtype=np.int64)

    train_groups, temp_groups = train_test_split(
        valid_groups,
        test_size=val_ratio + test_ratio,
        stratify=group_labels,
        random_state=random_state,
    )
    if test_ratio > 0:
        temp_labels = np.array([labels[group_ids == gid].max() for gid in temp_groups], dtype=np.int64)
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=relative_test,
            stratify=temp_labels,
            random_state=random_state,
        )
    else:
        val_groups, test_groups = temp_groups, np.array([], dtype=np.int64)

    train_idx = np.flatnonzero(np.isin(group_ids, train_groups) | (group_ids < 0))
    val_idx = np.flatnonzero(np.isin(group_ids, val_groups))
    test_idx = np.flatnonzero(np.isin(group_ids, test_groups))
    return train_idx, val_idx, test_idx


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
    sequences = data["sequences"]
    labels = data["labels"]
    sender_idx = data["sender_idx"]
    receiver_idx = data["receiver_idx"]
    alert_idx = data["alert_idx"]
    edge_attr = data["edge_attr"]
    sample_group_ids = data.get("split_group_ids", data["sample_group_ids"])

    train_idx, val_idx, test_idx = make_group_splits(
        sample_group_ids.numpy(),
        labels.numpy().astype(np.int64),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    if use_smote:
        data, train_idx = apply_smote_to_train(
            data, train_idx, smote_ratio=smote_ratio, random_state=random_state
        )
        sequences = data["sequences"]
        labels = data["labels"]
        sender_idx = data["sender_idx"]
        receiver_idx = data["receiver_idx"]
        alert_idx = data["alert_idx"]
        edge_attr = data["edge_attr"]

    dataset = TensorDataset(sequences, sender_idx, receiver_idx, edge_attr, alert_idx, labels)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    test_loader = None
    if len(test_idx) > 0:
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    metadata = {
        "schema_version": data.get("schema_version", 0),
        "split_unit": data.get("split_unit", "sender_account"),
        "group_type": data.get("group_type", "unknown"),
        "edge_feature_names": data.get("edge_feature_names", []),
        "n_types": data.get("n_types", 0),
        "n_accounts": data.get("n_accounts", 0),
        "n_groups": data.get("n_groups", 0),
        "num_samples": int(labels.size(0)),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "smote_applied": bool(data.get("smote_applied", False)),
        "original_train_size": int(data.get("original_train_size", len(train_idx))),
        "resampled_train_size": int(data.get("resampled_train_size", len(train_idx))),
    }
    return train_loader, val_loader, test_loader, metadata
