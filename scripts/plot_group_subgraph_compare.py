#!/usr/bin/env python3
"""
Plot "true group subgraph vs model high-score subgraph" for AML-style datasets.

Supported datasets:
- aml
- cryptopia
- amlsim_hi
- amlsim_li

Score file formats:
- node-level: columns include `node_idx` and `risk_score`
- sample-level: columns include `sample_idx` and `risk_score`

Optional score columns:
- `attention`: used to scale edge width
- `rule_text`: used as edge annotation

Optional edge trace columns:
- `attention`: edge attention exported by `edge_trace.csv`
- `matched_rule_text`: edge-level matched rule text
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true group subgraph vs model high-score subgraph.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["aml", "cryptopia", "amlsim_hi", "amlsim_li"],
        help="Formal dataset name.",
    )
    parser.add_argument(
        "--preprocessed_path",
        required=True,
        help="Path to preprocessed_*.pt produced by the project pipeline.",
    )
    parser.add_argument(
        "--score_file",
        required=True,
        help="CSV or JSON file containing node/sample risk scores.",
    )
    parser.add_argument(
        "--edge_trace_file",
        default=None,
        help="Optional edge_trace.csv exported by trainers. Used for edge attention and rule annotations.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. Defaults to figures/group_subgraph_compare_<dataset>.png beside the score file.",
    )
    parser.add_argument(
        "--group_id",
        type=int,
        default=None,
        help="Explicit group id to visualize.",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default=None,
        help="Explicit Cryptopia group name such as ml_transit_1.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k high-risk nodes or samples used to build the model subgraph.",
    )
    parser.add_argument(
        "--max_true_nodes",
        type=int,
        default=40,
        help="Limit the size of the true-group subgraph for readability.",
    )
    parser.add_argument(
        "--max_model_nodes",
        type=int,
        default=40,
        help="Limit the size of the high-score subgraph for readability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Layout seed.",
    )
    return parser.parse_args()


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_preprocessed(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        df = pd.DataFrame(payload)
    else:
        raise ValueError(f"Unsupported score file format: {ext}")
    return df


def load_scores(path: str) -> pd.DataFrame:
    df = _load_table(path)
    if "risk_score" not in df.columns:
        for fallback in ("score", "prob", "probability"):
            if fallback in df.columns:
                df = df.rename(columns={fallback: "risk_score"})
                break
    if "risk_score" not in df.columns:
        raise ValueError("score file must contain `risk_score` or an equivalent score column")
    return df


def detect_score_mode(df: pd.DataFrame) -> str:
    if "node_idx" in df.columns:
        return "node"
    if "sample_idx" in df.columns:
        return "sample"
    if "id" in df.columns:
        return "node"
    raise ValueError("score file must contain `node_idx`, `sample_idx`, or `id`")


def normalize_width(values: Iterable[float], base: float = 1.5, span: float = 4.0) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return []
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if abs(hi - lo) < 1e-8:
        return [base for _ in arr]
    scaled = (arr - lo) / (hi - lo)
    return list(base + span * scaled)


def build_graph_dataset_graph(data: Dict) -> Tuple[nx.Graph, Dict[int, int], Dict[int, float]]:
    edge_index = to_numpy(data["edge_index"])
    edge_raw_amount = to_numpy(data.get("edge_raw_amount"))
    group_ids = to_numpy(data.get("group_ids"))

    graph = nx.Graph()
    num_nodes = int(data["node_labels"].size(0))
    graph.add_nodes_from(range(num_nodes))

    if edge_raw_amount is None:
        edge_raw_amount = np.zeros(edge_index.shape[1], dtype=np.float32)

    for edge_id in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        graph.add_edge(src, dst, edge_id=edge_id, amount=float(edge_raw_amount[edge_id]))

    node_group_ids = {}
    if group_ids is not None:
        for idx, gid in enumerate(group_ids.tolist()):
            node_group_ids[int(idx)] = int(gid)

    return graph, node_group_ids, {int(i): 0.0 for i in range(num_nodes)}


def build_sample_dataset_graph(data: Dict) -> Tuple[nx.Graph, Dict[int, int], Dict[int, float]]:
    senders = to_numpy(data["sender_idx"]).astype(np.int64)
    receivers = to_numpy(data["receiver_idx"]).astype(np.int64)
    group_ids = to_numpy(data.get("group_ids"))
    raw_amounts = to_numpy(data.get("edge_raw_amount"))

    graph = nx.Graph()
    node_group_ids: Dict[int, int] = {}
    node_scores: Dict[int, float] = {}

    if raw_amounts is None:
        raw_amounts = np.zeros(len(senders), dtype=np.float32)

    for sample_idx, (src, dst) in enumerate(zip(senders, receivers)):
        src = int(src)
        dst = int(dst)
        graph.add_node(src)
        graph.add_node(dst)
        amount = float(raw_amounts[sample_idx])
        if graph.has_edge(src, dst):
            graph[src][dst]["sample_ids"].append(sample_idx)
            graph[src][dst]["amount"] = max(graph[src][dst]["amount"], amount)
        else:
            graph.add_edge(src, dst, sample_ids=[sample_idx], amount=amount)

        gid = int(group_ids[sample_idx]) if group_ids is not None else -1
        if gid >= 0:
            node_group_ids.setdefault(src, gid)
            node_group_ids.setdefault(dst, gid)
        node_scores.setdefault(src, 0.0)
        node_scores.setdefault(dst, 0.0)

    return graph, node_group_ids, node_scores


def select_group_id(
    dataset: str,
    data: Dict,
    node_group_ids: Dict[int, int],
    explicit_group_id: Optional[int],
    explicit_group_name: Optional[str],
) -> int:
    if explicit_group_id is not None:
        return explicit_group_id

    if explicit_group_name and "group_name_to_id" in data:
        mapping = data["group_name_to_id"]
        if explicit_group_name not in mapping:
            raise ValueError(f"group_name `{explicit_group_name}` not found in preprocessed data")
        return int(mapping[explicit_group_name])

    valid = [gid for gid in node_group_ids.values() if gid >= 0]
    if not valid:
        raise ValueError("no explicit group ids found in preprocessed data")

    counts = pd.Series(valid).value_counts()
    return int(counts.index[0])


def compute_node_risk_from_samples(
    graph: nx.Graph,
    score_df: pd.DataFrame,
) -> Dict[int, float]:
    node_risk: Dict[int, List[float]] = {int(n): [] for n in graph.nodes()}
    sample_col = "sample_idx" if "sample_idx" in score_df.columns else "id"
    score_lookup = {
        int(row[sample_col]): float(row["risk_score"])
        for _, row in score_df.iterrows()
    }

    for u, v, attrs in graph.edges(data=True):
        sample_ids = attrs.get("sample_ids", [])
        scores = [score_lookup[sid] for sid in sample_ids if sid in score_lookup]
        if not scores:
            continue
        edge_score = max(scores)
        node_risk[int(u)].append(edge_score)
        node_risk[int(v)].append(edge_score)

    return {
        node: (float(np.mean(scores)) if scores else 0.0)
        for node, scores in node_risk.items()
    }


def build_true_group_subgraph(
    graph: nx.Graph,
    node_group_ids: Dict[int, int],
    group_id: int,
    max_nodes: int,
) -> nx.Graph:
    true_nodes = [node for node, gid in node_group_ids.items() if gid == group_id]
    if not true_nodes:
        raise ValueError(f"group_id={group_id} has no nodes")

    subgraph = graph.subgraph(true_nodes).copy()
    if subgraph.number_of_edges() == 0:
        expanded_nodes = set(true_nodes)
        for node in true_nodes:
            expanded_nodes.update(graph.neighbors(node))
        subgraph = graph.subgraph(expanded_nodes).copy()

    if subgraph.number_of_nodes() > max_nodes:
        ranked = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = [node for node, _ in ranked]
        subgraph = subgraph.subgraph(keep).copy()

    return subgraph


def build_model_subgraph(
    dataset: str,
    graph: nx.Graph,
    score_df: pd.DataFrame,
    top_k: int,
    max_nodes: int,
) -> Tuple[nx.Graph, Dict[int, float]]:
    mode = detect_score_mode(score_df)

    if mode == "node":
        node_col = "node_idx" if "node_idx" in score_df.columns else "id"
        top_nodes = (
            score_df.sort_values("risk_score", ascending=False)[node_col]
            .astype(int)
            .head(top_k)
            .tolist()
        )
        model_nodes = set()
        for node in top_nodes:
            if node not in graph:
                continue
            model_nodes.add(node)
            model_nodes.update(graph.neighbors(node))
        if len(model_nodes) > max_nodes:
            ranked = sorted(
                [(n, graph.degree[n]) for n in model_nodes],
                key=lambda x: x[1],
                reverse=True,
            )[:max_nodes]
            model_nodes = {node for node, _ in ranked}
        subgraph = graph.subgraph(model_nodes).copy()
        node_risk = {
            int(row[node_col]): float(row["risk_score"])
            for _, row in score_df.iterrows()
            if int(row[node_col]) in subgraph
        }
        return subgraph, node_risk

    sample_col = "sample_idx" if "sample_idx" in score_df.columns else "id"
    top_samples = (
        score_df.sort_values("risk_score", ascending=False)[sample_col]
        .astype(int)
        .head(top_k)
        .tolist()
    )
    edge_nodes = set()
    for u, v, attrs in graph.edges(data=True):
        sample_ids = attrs.get("sample_ids", [])
        if any(sample_id in top_samples for sample_id in sample_ids):
            edge_nodes.add(int(u))
            edge_nodes.add(int(v))
    if len(edge_nodes) > max_nodes:
        ranked = sorted(
            [(n, graph.degree[n]) for n in edge_nodes],
            key=lambda x: x[1],
            reverse=True,
        )[:max_nodes]
        edge_nodes = {node for node, _ in ranked}
    subgraph = graph.subgraph(edge_nodes).copy()
    return subgraph, compute_node_risk_from_samples(subgraph, score_df)


def draw_subgraph(
    ax,
    subgraph: nx.Graph,
    title: str,
    node_risk: Dict[int, float],
    group_nodes: Optional[Iterable[int]] = None,
    score_df: Optional[pd.DataFrame] = None,
    edge_trace_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
):
    if subgraph.number_of_nodes() == 0:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, "Empty subgraph", ha="center", va="center")
        return

    pos = nx.spring_layout(subgraph, seed=seed)
    risk_values = np.array([node_risk.get(int(node), 0.0) for node in subgraph.nodes()], dtype=np.float32)

    if group_nodes is None:
        group_nodes = set()
    else:
        group_nodes = set(group_nodes)

    edge_width_values = []
    edge_labels = {}
    attention_lookup = {}
    rule_lookup = {}
    if score_df is not None and "sample_idx" in score_df.columns:
        if "attention" in score_df.columns:
            attention_lookup = {
                int(row["sample_idx"]): float(row["attention"])
                for _, row in score_df.iterrows()
            }
        if "rule_text" in score_df.columns:
            rule_lookup = {
                int(row["sample_idx"]): str(row["rule_text"])
                for _, row in score_df.iterrows()
            }
    edge_attention_lookup = {}
    edge_rule_lookup = {}
    if edge_trace_df is not None and {"src", "dst"}.issubset(edge_trace_df.columns):
        for _, row in edge_trace_df.iterrows():
            key = tuple(sorted((int(row["src"]), int(row["dst"]))))
            if "attention" in edge_trace_df.columns:
                edge_attention_lookup[key] = float(row.get("attention", 0.0))
            rule_text = str(row.get("matched_rule_text", "") or "")
            if rule_text:
                edge_rule_lookup[key] = rule_text

    for u, v, attrs in subgraph.edges(data=True):
        sample_ids = attrs.get("sample_ids", [])
        amount = float(attrs.get("amount", 0.0))
        width = amount
        edge_key = tuple(sorted((int(u), int(v))))
        if edge_key in edge_attention_lookup:
            width = edge_attention_lookup[edge_key]
        if sample_ids and attention_lookup:
            attn_scores = [attention_lookup[sid] for sid in sample_ids if sid in attention_lookup]
            if attn_scores:
                width = max(attn_scores)
        edge_width_values.append(width)

        if edge_key in edge_rule_lookup:
            edge_labels[(u, v)] = edge_rule_lookup[edge_key][:24]
            continue
        if sample_ids and rule_lookup:
            label_candidates = [rule_lookup[sid] for sid in sample_ids if sid in rule_lookup]
            if label_candidates:
                edge_labels[(u, v)] = label_candidates[0][:24]
                continue
        if amount:
            edge_labels[(u, v)] = f"{amount:.1f}"

    widths = normalize_width(edge_width_values)
    nx.draw_networkx_edges(
        subgraph,
        pos,
        ax=ax,
        width=widths if widths else 1.5,
        alpha=0.7,
        edge_color="#7a7a7a",
    )

    node_border = ["#111111" if int(node) in group_nodes else "#666666" for node in subgraph.nodes()]
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax,
        node_size=420,
        node_color=risk_values,
        cmap=plt.cm.Reds,
        linewidths=1.2,
        edgecolors=node_border,
    )
    nx.draw_networkx_labels(
        subgraph,
        pos,
        ax=ax,
        labels={node: str(node) for node in subgraph.nodes()},
        font_size=8,
    )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            subgraph,
            pos,
            ax=ax,
            edge_labels=edge_labels,
            font_size=7,
        )

    ax.set_title(title)
    ax.axis("off")


def resolve_output_path(score_file: str, output: Optional[str], dataset: str) -> str:
    if output:
        return output
    score_dir = os.path.dirname(os.path.abspath(score_file))
    figures_dir = os.path.join(score_dir, "..", "figures")
    figures_dir = os.path.abspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    return os.path.join(figures_dir, f"group_subgraph_compare_{dataset}.png")


def main():
    args = parse_args()
    data = load_preprocessed(args.preprocessed_path)
    score_df = load_scores(args.score_file)
    edge_trace_df = _load_table(args.edge_trace_file) if args.edge_trace_file else None

    if "edge_index" in data and "node_labels" in data:
        graph, node_group_ids, node_risk = build_graph_dataset_graph(data)
    else:
        graph, node_group_ids, node_risk = build_sample_dataset_graph(data)

    group_id = select_group_id(
        dataset=args.dataset,
        data=data,
        node_group_ids=node_group_ids,
        explicit_group_id=args.group_id,
        explicit_group_name=args.group_name,
    )

    true_subgraph = build_true_group_subgraph(
        graph=graph,
        node_group_ids=node_group_ids,
        group_id=group_id,
        max_nodes=args.max_true_nodes,
    )
    model_subgraph, model_node_risk = build_model_subgraph(
        dataset=args.dataset,
        graph=graph,
        score_df=score_df,
        top_k=args.top_k,
        max_nodes=args.max_model_nodes,
    )

    true_group_nodes = {node for node, gid in node_group_ids.items() if gid == group_id}
    if not node_risk or all(value == 0.0 for value in node_risk.values()):
        node_risk = model_node_risk

    output_path = resolve_output_path(args.score_file, args.output, args.dataset)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw_subgraph(
        axes[0],
        true_subgraph,
        title=f"True Group Subgraph (group_id={group_id})",
        node_risk=node_risk,
        group_nodes=true_group_nodes,
        score_df=score_df,
        edge_trace_df=edge_trace_df,
        seed=args.seed,
    )
    draw_subgraph(
        axes[1],
        model_subgraph,
        title=f"Model High-Score Subgraph (top_k={args.top_k})",
        node_risk=model_node_risk,
        group_nodes=true_group_nodes.intersection(model_subgraph.nodes()),
        score_df=score_df,
        edge_trace_df=edge_trace_df,
        seed=args.seed,
    )

    fig.suptitle(f"{args.dataset}: true group vs model high-score subgraph", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
