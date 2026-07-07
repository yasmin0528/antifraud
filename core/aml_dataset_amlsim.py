"""
AMLSIM tabular preprocessing for HI/LI variants with explicit pattern groups.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from core.aml_dataset import PREPROCESS_SCHEMA_VERSION


def _tx_key(
    timestamp: str,
    from_bank: str,
    from_account: str,
    to_bank: str,
    to_account: str,
    amount_paid: float,
    payment_format: str,
) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(timestamp),
        str(from_bank),
        str(from_account),
        str(to_bank),
        str(to_account),
        f"{float(amount_paid):.2f}",
        str(payment_format),
    )


def parse_pattern_groups(patterns_path: str) -> Tuple[Dict[Tuple[str, ...], int], Dict[int, str]]:
    tx_to_group: Dict[Tuple[str, ...], int] = {}
    group_type_map: Dict[int, str] = {}
    current_group = -1
    current_type = "background"

    with open(patterns_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                current_group += 1
                type_part = line.split(" - ", 1)[-1]
                current_type = type_part.split(":", 1)[0].strip()
                group_type_map[current_group] = current_type
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                continue
            key = _tx_key(
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                float(parts[7]),
                parts[9],
            )
            tx_to_group[key] = current_group

    return tx_to_group, group_type_map


def _resolve_account_columns(df: pd.DataFrame) -> Tuple[str, str]:
    from_account_col = "Account"
    to_account_col = "Account.1" if "Account.1" in df.columns else df.columns[4]
    return from_account_col, to_account_col


def preprocess_amlsim_data(
    trans_csv_path: str,
    patterns_path: str,
    window_size: int = 10,
    save_path: Optional[str] = None,
) -> Dict:
    df = pd.read_csv(trans_csv_path)
    from_account_col, to_account_col = _resolve_account_columns(df)

    tx_to_group, group_type_map = parse_pattern_groups(patterns_path)
    group_type_encoder = LabelEncoder()
    group_type_encoder.fit(list(group_type_map.values()) + ["background"])

    df["sender_account"] = (
        df["From Bank"].astype(str) + ":" + df[from_account_col].astype(str)
    )
    df["receiver_account"] = (
        df["To Bank"].astype(str) + ":" + df[to_account_col].astype(str)
    )
    df["payment_type"] = df["Payment Format"].astype(str)
    df["raw_amount"] = df["Amount Paid"].astype(float)
    df["label"] = df["Is Laundering"].astype(int)
    df["tx_key"] = [
        _tx_key(ts, fb, fa, tb, ta, amt, pf)
        for ts, fb, fa, tb, ta, amt, pf in zip(
            df["Timestamp"],
            df["From Bank"],
            df[from_account_col],
            df["To Bank"],
            df[to_account_col],
            df["Amount Paid"],
            df["Payment Format"],
        )
    ]
    df["group_id"] = df["tx_key"].map(tx_to_group).fillna(-1).astype(int)
    df["group_type"] = df["group_id"].map(group_type_map).fillna("background")
    df["group_type_id"] = group_type_encoder.transform(df["group_type"])

    df["timestamp_dt"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values(["sender_account", "timestamp_dt", "Timestamp"]).reset_index(drop=True)

    type_encoder = LabelEncoder()
    df["TX_TYPE_ID"] = type_encoder.fit_transform(df["payment_type"])

    all_accounts = pd.concat(
        [df["sender_account"], df["receiver_account"]], ignore_index=True
    )
    account_encoder = LabelEncoder()
    account_encoder.fit(all_accounts)
    df["SENDER_IDX"] = account_encoder.transform(df["sender_account"])
    df["RECEIVER_IDX"] = account_encoder.transform(df["receiver_account"])

    time_deltas = (
        df.groupby("sender_account")["timestamp_dt"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )
    df["TIME_DIFF"] = time_deltas.astype(float)

    amount_mean = float(df["raw_amount"].mean())
    amount_std = float(df["raw_amount"].std() + 1e-6)
    df["TX_AMOUNT_NORM"] = (df["raw_amount"] - amount_mean) / amount_std
    df["TX_LOG_AMOUNT"] = np.log1p(df["raw_amount"].clip(lower=0.0))

    n_types = int(df["TX_TYPE_ID"].nunique())
    n_accounts = int(account_encoder.classes_.shape[0])
    valid_group_ids = sorted(gid for gid in group_type_map.keys() if gid >= 0)
    n_groups = len(valid_group_ids)

    sequences: List[np.ndarray] = []
    labels_list: List[int] = []
    sender_indices: List[int] = []
    receiver_indices: List[int] = []
    pattern_indices: List[int] = []
    pattern_type_ids: List[int] = []
    split_group_ids: List[int] = []
    edge_attrs: List[List[float]] = []
    edge_raw_amounts: List[float] = []
    detection_features_list: List[List[float]] = []

    account_sequences = np.zeros((n_accounts, window_size, 3), dtype=np.float32)
    account_seq_len = np.zeros(n_accounts, dtype=np.int64)
    account_group_idx = np.full(n_accounts, -1, dtype=np.int64)
    pattern_to_accounts: Dict[int, list[int]] = {}
    pattern_to_samples: Dict[int, list[int]] = {}

    for _, group in df.groupby("sender_account"):
        group = group.reset_index(drop=True)
        sender_idx = int(group["SENDER_IDX"].iat[0])
        sender_seq = group[["TX_AMOUNT_NORM", "TX_TYPE_ID", "TIME_DIFF"]].to_numpy(
            dtype=np.float32
        )
        sender_recent = sender_seq[-window_size:]
        account_seq_len[sender_idx] = len(sender_recent)
        account_sequences[sender_idx, : len(sender_recent)] = sender_recent
        valid_groups = group.loc[group["group_id"] >= 0, "group_id"]
        if not valid_groups.empty:
            account_group_idx[sender_idx] = int(valid_groups.iat[-1])
            for group_id in sorted(set(valid_groups.astype(int).tolist())):
                pattern_to_accounts.setdefault(int(group_id), []).append(sender_idx)

        if len(group) < window_size:
            continue

        for i in range(len(group) - window_size + 1):
            window = group.iloc[i : i + window_size]
            features = window[["TX_AMOUNT_NORM", "TX_TYPE_ID", "TIME_DIFF"]].to_numpy(
                dtype=np.float32
            )
            last_row = window.iloc[-1]
            sequences.append(features)
            labels_list.append(int(last_row["label"]))
            sender_indices.append(int(last_row["SENDER_IDX"]))
            receiver_indices.append(int(last_row["RECEIVER_IDX"]))
            pattern_indices.append(int(last_row["group_id"]))
            pattern_type_ids.append(int(last_row["group_type_id"]))
            sample_idx = len(labels_list) - 1
            if pattern_indices[-1] >= 0:
                pattern_to_samples.setdefault(int(pattern_indices[-1]), []).append(sample_idx)
            split_gid = int(last_row["group_id"]) + n_accounts if int(last_row["group_id"]) >= 0 else int(last_row["SENDER_IDX"])
            split_group_ids.append(split_gid)
            edge_attrs.append(
                [
                    float(last_row["TX_AMOUNT_NORM"]),
                    float(last_row["TX_LOG_AMOUNT"]),
                    float(last_row["TIME_DIFF"]),
                ]
            )
            edge_raw_amounts.append(float(last_row["raw_amount"]))
            detection_features_list.append(
                [
                    float(last_row["TX_AMOUNT_NORM"]),
                    float(last_row["TX_TYPE_ID"]),
                    float(last_row["TIME_DIFF"]),
                ]
            )

    if not sequences:
        raise ValueError("No valid AMLSIM sequences created. Reduce window_size or inspect the dataset.")

    data = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "split_unit": "hybrid_pattern_sender",
        "group_type": "pattern_instance_id",
        "edge_feature_names": ["amount_norm", "log_amount", "time_diff"],
        "sample_group_ids": torch.tensor(np.array(sender_indices, dtype=np.int64), dtype=torch.long),
        "split_group_ids": torch.tensor(np.array(split_group_ids, dtype=np.int64), dtype=torch.long),
        "group_ids": torch.tensor(np.array(pattern_indices, dtype=np.int64), dtype=torch.long),
        "memory_group_ids": torch.tensor(np.array(pattern_indices, dtype=np.int64), dtype=torch.long),
        "group_labels": torch.tensor(np.array(pattern_type_ids, dtype=np.int64), dtype=torch.long),
        "sequences": torch.tensor(np.stack(sequences, axis=0), dtype=torch.float32),
        "labels": torch.tensor(np.array(labels_list, dtype=np.int64), dtype=torch.float32),
        "sender_idx": torch.tensor(np.array(sender_indices, dtype=np.int64), dtype=torch.long),
        "receiver_idx": torch.tensor(np.array(receiver_indices, dtype=np.int64), dtype=torch.long),
        "alert_idx": torch.tensor(np.array(pattern_indices, dtype=np.int64), dtype=torch.long),
        "edge_attr": torch.tensor(np.stack(edge_attrs, axis=0), dtype=torch.float32),
        "edge_raw_amount": torch.tensor(np.array(edge_raw_amounts, dtype=np.float32), dtype=torch.float32),
        "detection_features": torch.tensor(np.array(detection_features_list, dtype=np.float32), dtype=torch.float32),
        "account_sequences": torch.tensor(account_sequences, dtype=torch.float32),
        "account_seq_len": torch.tensor(account_seq_len, dtype=torch.long),
        "account_alert_idx": torch.tensor(account_group_idx, dtype=torch.long),
        "pattern_to_accounts": {int(k): sorted(set(v)) for k, v in pattern_to_accounts.items()},
        "pattern_to_samples": {int(k): sorted(set(v)) for k, v in pattern_to_samples.items()},
        "group_membership": {int(k): sorted(set(v)) for k, v in pattern_to_samples.items()},
        "amount_mean": amount_mean,
        "amount_std": amount_std,
        "n_types": n_types,
        "n_accounts": n_accounts,
        "n_groups": n_groups,
        "smote_applied": False,
        "group_type_names": group_type_map,
    }

    if save_path:
        torch.save(data, save_path)

    return data
