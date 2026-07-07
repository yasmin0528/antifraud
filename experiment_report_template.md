# Experiment Report Template

## 1. Experiment Overview

- Experiment name:
- Dataset:
- Variant:
- Seed:
- Run directory:
- Prediction granularity:
  - `sample-level` or `graph node-level`
- Explainability granularity:
  - `sample-level direct explanation`
  - or `edge-grounded node summary explanation`

## 2. Configuration

Recommended source files:

- `config.yaml`
- `results/metadata.json`

Suggested summary:

- Model pipeline: `CA1 -> CA3 -> MPFC -> VTA`
- Group type:
- Split unit:
- Memory mode:
- Whether alert/group metrics are enabled:
- Whether subgraph metrics are enabled:

## 3. Main Metrics

Recommended source:

- `results/results.csv`

Main table:

| Metric | Value |
|---|---:|
| AUC |  |
| F1 |  |
| AP |  |
| Best threshold |  |

## 4. Group/Subgraph Metrics

Use when available.

| Metric | Value |
|---|---:|
| Alert-level AP |  |
| Alert-level F1 |  |
| Hit@K |  |
| Subgraph coverage node |  |
| Subgraph coverage edge |  |

## 5. Explainability Output

### If the dataset is sample-level

Recommended files:

- `results/sample_scores.csv`
- `results/edge_trace.csv`

Suggested wording:

> Each prediction row corresponds to one transaction-window sample. The attached attention and rule fields are direct edge-grounded evidence for that sample.

### If the dataset is graph node-level

Recommended files:

- `results/node_scores.csv`
- `results/edge_trace.csv`

Suggested wording:

> Each prediction row corresponds to one graph node. Node-level explanation fields are summaries aggregated from incident edge evidence. The strict structural evidence remains in `edge_trace.csv`.

## 6. Case Study

Recommended structure:

1. Pick one high-risk sample or node
2. Report its `risk_score`
3. Report its `group_id`
4. Report the most relevant `rule_text / rule_type / rule_confidence`
5. If needed, trace the supporting edge rows from `edge_trace.csv`

Case notes:

- Selected entity:
- Risk score:
- Group id:
- Key rule text:
- Key rule type:
- Rule confidence:
- Supporting edge ids:

## 7. Visualization

Recommended figure:

- `figures/group_subgraph_compare_<dataset>.png`

Suggested description:

> The left panel shows the true group subgraph, while the right panel shows the model high-score subgraph. Edge widths represent attention, and edge annotations summarize matched rules when available.

## 8. Interpretation

Suggested questions to answer:

1. Did the model capture the correct ranking signal?
2. Did it recover the true group or suspicious subgraph structure?
3. Is the explanation directly attached to the prediction, or aggregated from edge evidence?
4. Are the gains mainly from sequence modeling, graph reasoning, or rule guidance?

## 9. Limitations

Suggested wording options:

- `CA3` is currently an explicit group memory enhancer, not a full case retrieval system.
- `VTA` is currently a training-time modulation mechanism, not a fully developed feedback controller.
- For graph node-level tasks, explanation is edge-grounded and then aggregated to nodes.
- The LLM currently acts as a rule generator rather than a strictly verified symbolic reasoner.

## 10. Final Conclusion

Suggested one-paragraph close:

> This run verifies the current implementation under the specified dataset and variant. The reported performance should be interpreted together with the prediction granularity and explanation granularity. For sample-level datasets, explanation is directly attached to samples; for graph node-level datasets, explanation is derived from edge-grounded evidence and summarized at the node level.
