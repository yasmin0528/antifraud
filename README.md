# AML Anti-Money Laundering Project

本项目当前的正式实验实现统一为 `CA1 -> CA3 -> MPFC -> VTA`。
代码已经支持四个正式数据入口：

- `aml`
- `cryptopia`
- `amlsim_hi`
- `amlsim_li`

其中：

- `aml` 和 `amlsim_*` 走 sample-level 管线，监督单元是交易窗口样本。
- `cryptopia` 走 graph node-level 管线，监督单元是地址节点。
- `cryptopia_graph` 只保留兼容含义，不应再作为论文中的正式数据集名称。

## 1. Current Implementation Status

当前代码已经落地的能力：

- `CA1` 负责账户/节点局部时序编码，输出 embedding 与 `score_micro`。
- `CA3` 已升级为显式 group memory 版本，支持按 `ALERT_ID`、`pattern_instance_id` 或 `ml_transit_*` group 进行聚合更新。
- `MPFC` 已支持 rule-guided GAT、task gating、LLM 规则导入、边级 rule trace 导出。
- `VTA` 已支持训练期 `da_signal` 调制，但仍属于轻量 neuromodulation，不是严格反馈控制器。
- 已支持 group-level 评估指标：`alert_level_ap`、`alert_level_f1`、`hit_at_k`、`subgraph_coverage_*`。
- 已支持可视化脚本读取 `score_file + edge_trace.csv` 进行子图对比分析。

当前仍然属于“已实现第一版、但不是最终论文完结版”的部分：

- `CA3` 虽然已经使用显式 group 元信息更新 memory，但还不是完整的跨案件案例库检索系统。
- `graph node` 任务的解释本质上仍然来自边级证据，再聚合为节点摘要，不应表述成“原生节点解释器”。
- `LLM` 当前承担“规则生成器”角色，不应表述成严格可验证的符号推理器。

## 2. Dataset Mapping

### `aml`

- 原始数据：`AMLdataset.csv`
- 监督单元：sample-level 交易窗口
- split 单元：`SENDER_ACCOUNT_ID`
- 显式 group：`ALERT_ID`
- 适合验证：全链路时序 + 账户图 + 告警级评估

### `cryptopia`

- 原始数据：`CryptopiaHacker/`
- 监督单元：graph node-level 地址节点
- split 单元：`train_mask / val_mask / test_mask`
- 显式 group：地址标签中的 `ml_transit_*`
- 适合验证：图结构传播、黑产地址识别、子图恢复

### `amlsim_hi`

- 原始数据：
  - `AMLSIM/HI-Small_Trans.csv`
  - `AMLSIM/HI-Small_accounts.csv`
  - `AMLSIM/HI-Small_Patterns.txt`
- 监督单元：sample-level 交易窗口
- 显式 group：从 `Patterns.txt` 解析出的 `pattern_instance_id`
- 适合验证：pattern/group 结构迁移

### `amlsim_li`

- 原始数据：
  - `AMLSIM/LI-Small_Trans.csv`
  - `AMLSIM/LI-Small_accounts.csv`
  - `AMLSIM/LI-Small_Patterns.txt`
- 监督单元：sample-level 交易窗口
- 显式 group：从 `Patterns.txt` 解析出的 `pattern_instance_id`
- 适合验证：低复杂度 pattern 数据上的稳定性

## 3. Explainability Modes

这是当前文档必须明确区分的部分。

### Sample-Level Explainability

适用数据集：

- `aml`
- `amlsim_hi`
- `amlsim_li`

预测对象：

- 一条 `sample`，即一个交易滑窗样本

导出文件：

- `results/sample_scores.csv`
- `results/edge_trace.csv`

解释语义：

- `sample_scores.csv` 中每一行对应一个样本。
- 当前 sample-level 管线中，“样本”和“批图中的一条边”是一一对应的，因此可以把边级解释直接回填到样本行。
- 这里的 `attention / rule_text / rule_type / rule_confidence` 可以被解释为该样本对应交易边的直接证据。

适合写进论文的表述：

> 在 sample-level 数据集上，模型对每个交易窗口样本输出风险分数，并同时导出该样本对应交易边的注意力与匹配规则，形成直接样本级解释。

### Graph Node Explainability

适用数据集：

- `cryptopia`

预测对象：

- 一个地址节点 `node`

导出文件：

- `results/node_scores.csv`
- `results/edge_trace.csv`

解释语义：

- `node_scores.csv` 中每一行对应一个节点预测结果。
- 但 `MPFC` 的原生解释信号仍然产生在边上，因此 `node_scores.csv` 里的解释字段是由该节点 incident edges 聚合得到的摘要：
  - `attention`：邻接边 attention 的聚合值
  - `rule_text / rule_type / rule_confidence`：最高注意力边对应的规则摘要
- 如果需要严格证据链，应以 `edge_trace.csv` 为准，而不是把 `node_scores.csv` 当成逐节点原生规则证明。

适合写进论文的表述：

> 在 graph node-level 数据集上，模型首先产生边级结构解释，再将邻接边证据聚合为节点级摘要；因此节点解释属于“边证据聚合后的节点摘要”，而非独立生成的原生节点解释。

### Practical Recommendation

如果你在写实验报告或论文：

- `aml / amlsim_*`：可以写“sample-level direct explanation”
- `cryptopia`：应写“edge-grounded node summary explanation”

不要混写成统一的“instance-level explanation”，否则语义不严谨。

## 4. Environment

建议拆分为两个环境：

```bash
conda create -n llm python=3.10 -y
conda create -n antifraud python=3.10 -y

conda activate llm
pip install -r requirements_llm.txt

conda activate antifraud
pip install -r requirements_antifraud.txt
```

## 5. Common Commands

默认在 `refactored/` 目录下运行。

### Single Run

```bash
python train.py --config configs/default.yaml --dataset aml
python train.py --config configs/dataset/cryptopia.yaml
python train.py --config configs/dataset/amlsim_hi.yaml
python train.py --config configs/dataset/amlsim_li.yaml
```

### Ablation

```bash
python train.py --config configs/ablation/wo_ca3.yaml --dataset aml
python train.py --config configs/ablation/wo_vta.yaml --config configs/dataset/cryptopia.yaml
```

### Batch Experiments

```bash
python scripts/run_experiments.py --only ablation
python scripts/run_experiments.py --only sensitivity
python scripts/run_experiments.py --only multi_seed
```

### Smoke Training

Use these to verify that the training pipeline, evaluation, and result export all run end-to-end before starting long experiments.

```bash
python train.py --config configs/default.yaml --config configs/dataset/cryptopia.yaml --config configs/smoke/cryptopia.yaml
python train.py --config configs/default.yaml --config configs/dataset/aml.yaml --config configs/smoke/aml.yaml
```

Repository checklist:

- `refactored/smoke_training_checklist.md`

## 6. Output Convention

每次运行输出到：

```text
outputs/<dataset>/<experiment_name>/run_YYYYMMDD_HHMMSS/
```

标准目录结构：

```text
config.yaml
log/
figures/
ckpt/
tensorboard/
results/
```

`results/` 当前建议视为论文与报告的主取数目录，常见文件包括：

```text
results/results.csv
results/metadata.json
results/ca3_memory.pt
results/ca3_memory_stats.json
results/group_meta.json
results/sample_scores.csv        # sample-level datasets
results/node_scores.csv          # graph node-level datasets
results/edge_trace.csv
```

如果启用了 LLM 规则更新，运行目录下还会有：

```text
llm_rules.json
```

## 7. Result File Semantics

### `results/results.csv`

这是主汇总表，适合直接做论文表格。

当前核心字段包括：

- `dataset`
- `variant`
- `seed`
- `auc`
- `f1`
- `ap`
- `best_threshold`
- `train_size`
- `val_size`
- `test_size`
- `smote_applied`
- `alert_level_ap`
- `alert_level_f1`
- `hit_at_k`
- `subgraph_coverage_node`
- `subgraph_coverage_edge`

建议：

- 主文表格保留 `auc / f1 / ap`
- group/subgraph 指标可放正文补充表或附录

### `results/metadata.json`

适合记录一次 run 的上下文，当前通常包含：

- 数据集名称与路径
- `schema_version`
- `split_unit`
- `group_type`
- `memory_mode`
- train/val/test 大小
- 是否启用 group/subgraph eval

### `results/sample_scores.csv`

适用于 `aml`、`amlsim_hi`、`amlsim_li`。

当前字段：

- `sample_idx`
- `sender_idx`
- `receiver_idx`
- `group_id`
- `label`
- `risk_score`
- `attention`
- `rule_text`
- `rule_type`
- `rule_confidence`

用途：

- 直接做样本级排序分析
- 结合 `group_id` 做 alert/pattern 级统计
- 作为 sample-level explainability 主表

### `results/node_scores.csv`

适用于 `cryptopia`。

当前字段：

- `node_idx`
- `group_id`
- `label`
- `risk_score`
- `attention`
- `rule_text`
- `rule_type`
- `rule_confidence`

注意：

- 这里的解释字段是节点邻接边证据聚合后的摘要，不是节点原生 rule trace。
- 如果要追具体结构证据，请联查 `edge_trace.csv`。

### `results/edge_trace.csv`

这是当前最严格的结构解释导出文件。

字段：

- `edge_id`
- `src`
- `dst`
- `attention`
- `rule_bias`
- `rule_match_score`
- `matched_rule_text`
- `matched_rule_type`
- `rule_confidence`

用途：

- 解释某条边为什么被高亮
- 支撑子图可视化中的边宽和边注释
- 作为 graph node explainability 的底层证据

### `results/group_meta.json`

这是 `CA3` 显式 group memory 的辅助导出，适合说明 memory slot 与 group 的对应关系。

### `results/ca3_memory.pt` and `results/ca3_memory_stats.json`

用于保留 `CA3` memory state，方便后验分析和复现实验。

## 8. Visualization

当前保留的可视化方案是：

- 左图：真实 group 子图
- 右图：模型高分子图

示例：

```bash
python scripts/plot_group_subgraph_compare.py \
  --dataset cryptopia \
  --preprocessed_path preprocessed_cryptopia.pt \
  --score_file outputs/cryptopia/baseline/run_xxx/results/node_scores.csv \
  --edge_trace_file outputs/cryptopia/baseline/run_xxx/results/edge_trace.csv
```

脚本支持：

- 从预处理缓存恢复图或样本图
- 自动或手动指定真实 group
- 读取 `node_scores.csv` 或 `sample_scores.csv`
- 读取 `edge_trace.csv` 叠加边 attention 与规则文本

推荐做法：

- `aml / amlsim_*` 使用 `sample_scores.csv + edge_trace.csv`
- `cryptopia` 使用 `node_scores.csv + edge_trace.csv`

## 9. Recommended Reporting Format

如果要直接写实验报告，建议最少保留以下结构：

1. 实验设置
2. 主指标表：`auc / f1 / ap`
3. group/subgraph 补充指标表：`alert_level_ap / alert_level_f1 / hit_at_k / subgraph_coverage_*`
4. 一张子图对比图
5. 一个 explainability 案例：
   - sample-level 数据集：引用 `sample_scores.csv`
   - graph node-level 数据集：引用 `node_scores.csv`，并附 `edge_trace.csv` 证据

建议在报告正文中明确写出：

- 预测粒度是什么
- 解释粒度是什么
- 两者是否一致

可直接使用的模板文件：

- `refactored/experiment_report_template.md`

## 10. Wording Rules For Paper Writing

建议统一使用以下口径：

- `aml / amlsim_*`：sample-level risk detection with direct edge-grounded explanation
- `cryptopia`：graph node-level risk detection with edge-grounded node summary explanation
- `CA3`：explicit group memory enhancement
- `VTA`：training-time neuromodulation

不建议使用以下表述：

- “CA3 已完成跨案件记忆库检索”
- “Cryptopia 具有原生节点规则解释”
- “LLM 已完成严格符号推理”

## 11. Migration Note

以下旧口径不再作为正式论文表述：

- 旧的 `Cryptopia` 表格滑窗链路
- `cryptopia_graph` 作为正式数据集名
- 把所有解释统一写成同一种 instance-level explanation

后续所有图表、表格、实验报告，建议统一以本 README 的术语为准。
