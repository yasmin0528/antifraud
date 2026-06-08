"""
CryptopiaHacker 全量地址特征工程。

核心思路：
- 使用 all-address.csv (280K) 获取标签
- 使用 all-normal-address.csv (152K) + all-normal-tx-cube_xy.csv 构建交易特征
- 对每个地址提取发送方/接收方统计、时序、网络、金额分布特征
- 输出: feature matrix + train/val/test split
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_cryptopia_features(
    data_dir: str = "/mnt/e/反洗钱/CryptopiaHacker",
    save_path: Optional[str] = "cryptopia_features.npz",
) -> Dict:
    """
    构建 CryptopiaHacker 全量地址特征。

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_cols, trn_idx, val_idx, tst_idx
    """
    # ================================================================
    # 1. 加载地址和标签
    # ================================================================
    print("[1/6] Loading addresses...")
    all_addr = pd.read_csv(os.path.join(data_dir, "all-address.csv"))
    all_addr["is_heist"] = (all_addr["label"] == "heist").astype(float)

    # 建立 address -> all-address index 映射
    addr_to_idx = {addr: i for i, addr in enumerate(all_addr["address"])}

    # ================================================================
    # 2. 映射 normal 地址索引
    # ================================================================
    print("[2/6] Mapping normal addresses to all-address indices...")
    norm_addr = pd.read_csv(os.path.join(data_dir, "all-normal-address.csv"))

    norm_records = []
    for i, row in norm_addr.iterrows():
        addr = row["address"]
        if addr in addr_to_idx:
            norm_records.append(
                {
                    "norm_idx": i,
                    "all_idx": addr_to_idx[addr],
                    "is_heist": 1.0 if row["label"] == "heist" else 0.0,
                }
            )

    norm_df = pd.DataFrame(norm_records)
    n_total = len(norm_records)
    print(f"  Mapped {n_total}/{len(norm_addr)} normal addresses")

    # ================================================================
    # 3. 加载交易 cube_xy
    # ================================================================
    print("[3/6] Loading transactions (cube_xy)...")
    cube = pd.read_csv(
        os.path.join(data_dir, "all-normal-tx-cube_xy.csv"), header=None
    )
    cube.columns = ["sender_norm_idx", "receiver_norm_idx", "timestamp", "amount"]
    print(f"  Transactions: {len(cube):,}")

    # ================================================================
    # 4. 分组统计：发送方特征
    # ================================================================
    print("[4/6] Computing sender features...")
    sender_agg = (
        cube.groupby("sender_norm_idx")
        .agg(
            send_count=("amount", "count"),
            send_sum=("amount", "sum"),
            send_mean=("amount", "mean"),
            send_std=("amount", "std"),
            send_max=("amount", "max"),
            send_min=("amount", "min"),
            send_first_ts=("timestamp", "min"),
            send_last_ts=("timestamp", "max"),
            send_unique_rcv=("receiver_norm_idx", "nunique"),
        )
        .reset_index()
    )
    # Rename index column explicitly
    sender_agg.rename(columns={"sender_norm_idx": "norm_idx"}, inplace=True)

    # ================================================================
    # 5. 分组统计：接收方特征
    # ================================================================
    print("[5/6] Computing receiver features...")
    receiver_agg = (
        cube.groupby("receiver_norm_idx")
        .agg(
            recv_count=("amount", "count"),
            recv_sum=("amount", "sum"),
            recv_mean=("amount", "mean"),
            recv_std=("amount", "std"),
            recv_max=("amount", "max"),
            recv_min=("amount", "min"),
            recv_first_ts=("timestamp", "min"),
            recv_last_ts=("timestamp", "max"),
            recv_unique_snd=("sender_norm_idx", "nunique"),
        )
        .reset_index()
    )
    receiver_agg.rename(columns={"receiver_norm_idx": "norm_idx"}, inplace=True)

    # ================================================================
    # 6. 合并特征
    # ================================================================
    print("[6/6] Merging and computing derived features...")

    # 将 norm_idx 转为整数
    features_df = norm_df[["norm_idx", "all_idx", "is_heist"]].copy()
    features_df["norm_idx"] = features_df["norm_idx"].astype(int)

    # merge with sender stats
    features_df = features_df.merge(
        sender_agg, on="norm_idx", how="left"
    )

    # merge with receiver stats
    features_df = features_df.merge(
        receiver_agg, on="norm_idx", how="left"
    )

    # 填充空值（无发送/接收的地址）
    fill_cols = [
        "send_count",
        "send_sum",
        "send_mean",
        "send_std",
        "send_max",
        "send_min",
        "send_first_ts",
        "send_last_ts",
        "send_unique_rcv",
        "recv_count",
        "recv_sum",
        "recv_mean",
        "recv_std",
        "recv_max",
        "recv_min",
        "recv_first_ts",
        "recv_last_ts",
        "recv_unique_snd",
    ]
    for col in fill_cols:
        features_df[col] = features_df[col].fillna(0.0).astype(float)

    # ---- 收发比 ----
    features_df["send_recv_ratio"] = features_df["send_count"] / (
        features_df["recv_count"] + 1
    )
    features_df["send_recv_amount_ratio"] = features_df["send_sum"] / (
        features_df["recv_sum"] + 1
    )

    # ---- 总交易量 ----
    features_df["total_tx"] = features_df["send_count"] + features_df["recv_count"]
    features_df["total_amount"] = features_df["send_sum"] + features_df["recv_sum"]

    # ---- 时间特征 ----
    features_df["time_span_send"] = (
        features_df["send_last_ts"] - features_df["send_first_ts"]
    )
    features_df["time_span_recv"] = (
        features_df["recv_last_ts"] - features_df["recv_first_ts"]
    )
    features_df["tx_frequency_send"] = features_df["send_count"] / (
        features_df["time_span_send"] / 86400 + 1
    )
    features_df["tx_frequency_recv"] = features_df["recv_count"] / (
        features_df["time_span_recv"] / 86400 + 1
    )

    # ---- 金额集中度 ----
    features_df["send_max_ratio"] = features_df["send_max"] / (
        features_df["send_sum"] + 1
    )
    features_df["recv_max_ratio"] = features_df["recv_max"] / (
        features_df["recv_sum"] + 1
    )

    # ---- 网络复杂度 ----
    features_df["total_unique_counterparties"] = (
        features_df["send_unique_rcv"] + features_df["recv_unique_snd"]
    )
    features_df["avg_tx_per_counterparty_send"] = features_df["send_count"] / (
        features_df["send_unique_rcv"] + 1
    )
    features_df["avg_tx_per_counterparty_recv"] = features_df["recv_count"] / (
        features_df["recv_unique_snd"] + 1
    )

    # ---- 对数变换 ----
    for col in [
        "send_sum",
        "recv_sum",
        "total_amount",
        "send_count",
        "recv_count",
        "total_tx",
    ]:
        features_df[f"log_{col}"] = np.log1p(features_df[col])

    # --- 大额交易比例 ----
    features_df["send_large_tx_ratio"] = 0.0
    features_df["send_above_mean_ratio"] = 0.0

    # 分批处理有发送交易的地址
    send_mask = features_df["send_count"] > 0
    send_indices = features_df[send_mask].index
    for idx in send_indices:
        nidx = int(features_df.loc[idx, "norm_idx"])
        addr_txs = cube[cube["sender_norm_idx"] == nidx]
        if len(addr_txs) > 0:
            amounts = addr_txs["amount"].values
            threshold = float(amounts.mean() + 2.0 * amounts.std())
            features_df.loc[idx, "send_large_tx_ratio"] = float(
                np.sum(amounts > threshold) / len(amounts)
            )
            features_df.loc[idx, "send_above_mean_ratio"] = float(
                np.mean(amounts > amounts.mean())
            )

    # ---- 特征列 ----
    feature_cols = [c for c in features_df.columns if c not in ["norm_idx", "all_idx", "is_heist"]]
    print(f"  Feature matrix: {features_df.shape}")
    print(f"  Heist rate: {features_df['is_heist'].mean():.4f} ({int(features_df['is_heist'].sum())}/{len(features_df)})")
    print(f"  Features: {len(feature_cols)}")

    # ================================================================
    # 7. 划分 train/val/test
    # ================================================================
    print("\nSplitting train/val/test...")
    X = features_df[feature_cols].values.astype(np.float64)
    y = features_df["is_heist"].values.astype(np.float64)

    trn_idx, tmp_idx = train_test_split(
        np.arange(len(features_df)), test_size=0.2, stratify=y, random_state=42
    )
    val_idx, tst_idx = train_test_split(
        tmp_idx, test_size=0.5, stratify=y[tmp_idx], random_state=42
    )

    X_train, X_val, X_test = X[trn_idx], X[val_idx], X[tst_idx]
    y_train, y_val, y_test = y[trn_idx], y[val_idx], y[tst_idx]

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"  Train: {len(trn_idx)}, heist={y_train.mean():.4f}")
    print(f"  Val:   {len(val_idx)}, heist={y_val.mean():.4f}")
    print(f"  Test:  {len(tst_idx)}, heist={y_test.mean():.4f}")

    # ================================================================
    # 8. 保存
    # ================================================================
    data = {
        "X_train": X_train_s,
        "y_train": y_train,
        "X_val": X_val_s,
        "y_val": y_val,
        "X_test": X_test_s,
        "y_test": y_test,
        "trn_idx": trn_idx,
        "val_idx": val_idx,
        "tst_idx": tst_idx,
        "feature_cols": np.array(feature_cols, dtype=object),
    }

    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved to {save_path}")

    return data


if __name__ == "__main__":
    build_cryptopia_features()
