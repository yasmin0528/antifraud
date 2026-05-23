# AML 反洗钱检测模型

基于海马-前额叶环路（CA1 ⟶ CA3 ⟶ mPFC ⟶ VTA）的图神经网络反洗钱交易检测模型。

## 环境配置

依赖包较多，建议拆分为两个 conda 环境，避免 vLLM、xformers、PyTorch Geometric 冲突：

```bash
conda create -n llm python=3.10 -y
conda create -n antifraud python=3.10 -y

# LLM 环境（仅运行 LLM 服务用）
conda activate llm
pip install -r requirements_llm.txt

# 模型环境（训练和推理）
conda activate antifraud
pip install -r requirements_antifraud.txt
```

项目根目录下操作：

```bash
cd /path/to/antifraud

# 确保 train.py 有执行权限
chmod +x train.py
```

## 目录结构

```
.
├── train.py                          # 统一训练入口
├── scripts/
│   ├── run_experiments.py            # 实验编排器（批量运行各类实验）
│   └── view_llm_rules.py             # 查看 LLM 生成的规则
├── core/                             # 核心模块（数据集、训练器、编排器）
├── configs/
│   ├── default.yaml                  # 默认配置（完整模型）
│   ├── ablation/                     # 消融实验配置（wo_*）
│   ├── sensitivity/                  # 参数敏感性配置
│   └── vta_decomposition.yaml        # VTA 损失解耦配置
├── models/                           # 模型组件
└── utils/                            # 工具（日志/checkpoint/可视化）
```

## 实验清单

| 实验类型 | 说明 | 子实验数 | 预计总时长 |
|----------|------|----------|-----------|
| baseline | 完整 MPFC 模型 | 1 | ~30min |
| ablation | 逐模块消融（CA1/CA3/MPFC/VTA/LLM） | 5 | 1h+1.5h+4.5h |
| sensitivity | 参数敏感性（lr/focal_gamma/rpe_beta/memory_momentum） | 4 | ~2h |
| vta_decomp | VTA 损失解耦（Focal/RPE/PW 逐组件清零） | 5 | ~2.5h |
| multi_seed | 多随机种子稳定性评估（5 seeds） | 5 | ~2.5h |

> 以上时长按 `--epochs 10`、单 GPU 估计。

## 通用参数

所有实验共用以下参数：

| 参数 | 说明 | 必需 |
|------|------|------|
| `--llm_api_url <URL>` | LLM API 端点 | 需要 LLM 的实验必填 |
| `--data_path <PATH>` | 数据集 CSV 路径 | 需要数据的实验必填 |
| `--preprocessed_path <PATH>` | 预处理数据路径（可选） | 否 |
| `--epochs <N>` | 训练轮数（默认 10） | 否 |
| `--output_dir <DIR>` | 输出目录（默认 outputs/） | 否 |

---

## 一、主实验（Baseline）

运行完整 MPFC 模型（含 LLM 规则生成）：

```bash
# 方式一：直接通过 train.py 运行
python train.py \
    --config configs/default.yaml \
    --llm_api_url http://127.0.0.1:23333/v1/chat/completions \
    --name baseline \
    --epochs 10
```

输出目录：`outputs/baseline/run_YYYYMMDD_HHMMSS/`

### 断点续训

```bash
# 自动恢复 outputs/baseline/ 下最新的 run_* 目录
python train.py --resume

# 指定 checkpoint 恢复
python train.py --resume outputs/baseline/run_20260523_120000/checkpoint/latest.pt
```

### 测试已训练的模型

```bash
python train.py \
    --config configs/default.yaml \
    --test \
    --checkpoint outputs/baseline/run_20260523_120000/checkpoint/best.pt
```

---

## 二、消融实验（Ablation Study）

验证各模块贡献度。共 5 个变体，每个独立运行：

| 变体 | 配置 | 说明 |
|------|------|------|
| wo_ca1 | `configs/ablation/wo_ca1.yaml` | 移除时序序列编码（CA1） |
| wo_ca3 | `configs/ablation/wo_ca3.yaml` | 移除关联记忆增强（CA3） |
| wo_mpfc | `configs/ablation/wo_mpfc.yaml` | 移除图推理模块，用线性分类器替代 |
| wo_vta | `configs/ablation/wo_vta.yaml` | 用标准 BCE Loss 替代 VTA 加权损失 |
| wo_llm | `configs/ablation/wo_llm.yaml` | 保留 mPFC 图架构，去掉 LLM 符号规则 |

### 批量运行（推荐）

```bash
 python scripts/run_experiments.py \
     --only ablation \
     --llm_api_url http://127.0.0.1:23333/v1/chat/completions
```

自动扫描 `configs/ablation/*.yaml`，逐一执行所有变体。

### 单独运行某个变体

```bash
python train.py \
    --config configs/ablation/wo_ca1.yaml \
    --llm_api_url http://127.0.0.1:23333/v1/chat/completions \
    --name ablation_wo_ca1 \
    --epochs 50
```

输出目录：
```
outputs/ablation_wo_ca1/run_20260523_120001/
├── log/<exp_name>.log
├── figures/{loss_curve,roc_curve,pr_curve,confusion_matrix}.png
├── checkpoint/{latest,best}.pt
├── tensorboard/events.out...
└── results/results.csv
```

---

## 三、参数敏感性实验（Sensitivity Analysis）

网格搜索关键超参数。配置定义在 `configs/sensitivity/` 目录下，自动扫描。

### 批量运行

```bash
python scripts/run_experiments.py \
    --only sensitivity \
    --llm_api_url http://127.0.0.1:23333/v1/chat/completions \
```

### 当前扫描的参数

| 参数 | 文件 | 扫描范围 |
|------|------|----------|
| learning_rate | `configs/sensitivity/learning_rate.yaml` | [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] |
| focal_gamma | `configs/sensitivity/focal_gamma.yaml` | [0.0, 0.5, 1.0, 2.0, 4.0] |
| rpe_beta | `configs/sensitivity/rpe_beta.yaml` | [0.0, 0.5, 1.0, 1.5, 3.0] |
| memory_momentum | `configs/sensitivity/memory_momentum.yaml` | [0.0, 0.5, 0.7, 0.9, 0.99] |

### 单独运行某个扫描

```bash
python train.py \
    --config configs/sensitivity/learning_rate.yaml \
    --llm_api_url http://127.0.0.1:23333/v1/chat/completions \
    --name sensitivity_learning_rate \
    --epochs 10
```

---

## 四、创新实验（VTA 损失解耦 + 多种子稳定性）

### VTA 损失解耦

验证损失函数各组件的贡献。遍历 5 个变体：

| 变体 | focal_gamma | rpe_beta | pos_weight | 说明 |
|------|-------------|----------|------------|------|
| full | 2.0 | 1.5 | 5.0 | 完整 VTA 损失 |
| wo_focal | 0.0 | 1.5 | 5.0 | 去掉焦点损失 |
| wo_rpe | 2.0 | 0.0 | 5.0 | 去掉奖励预测误差 |
| wo_pw | 2.0 | 1.5 | 1.0 | 去掉正样本加权 |
| bce_only | 0.0 | 0.0 | 1.0 | 仅标准 BCE |

```bash
python scripts/run_experiments.py \
    --only vta_decomp \
    --data_path /path/to/dataset.csv
```

> VTA 解耦实验不依赖 LLM，无需 `--llm_api_url`。

### 多随机种子实验

评估模型在不同随机种子下的稳定性（默认 5 个种子：42, 123, 456, 789, 1111）：

```bash
python scripts/run_experiments.py \
    --only multi_seed \
    --llm_api_url http://127.0.0.1:23333/v1/chat/completions \
    --data_path /path/to/dataset.csv
```

种子列表从 `configs/default.yaml` 的 `multi_seed.seeds` 读取，可通过修改配置文件调整。

---

## 五、一键运行全部实验

```bash
python scripts/run_experiments.py \
    --llm_api_url http://127.0.0.1:23333/v1/chat/completions \
    --data_path /path/to/dataset.csv \
    --epochs 10
```

依次执行：baseline → ablation → sensitivity → vta_decomp → multi_seed，完成后自动汇总结果。

> `scripts/run_experiments.py` 不接受 `--config` 参数。配置文件路径由脚本内部自动拼接。

### 常用选项

| 参数 | 说明 |
|------|------|
| `--only <type>` | 仅运行指定类型实验（baseline/ablation/sensitivity/vta_decomp/multi_seed） |
| `--dry-run` | 只打印命令，不实际执行 |
| `--summarize-only` | 仅汇总已有结果，不重新运行 |
| `--no-summarize` | 跳过最终汇总 |
| `--export <path>` | 导出汇总结果到 CSV 文件 |

---

## 输出目录结构

一次实验 = 唯一一个 `run_*` 文件夹：

```
outputs/<exp_name>/run_YYYYMMDD_HHMMSS/
├── log/<exp_name>.log                # 文本日志
├── figures/
│   ├── loss_curve.png                # 训练 Loss 曲线
│   ├── roc_curve.png                 # ROC 曲线（含 AUC）
│   ├── pr_curve.png                  # Precision-Recall 曲线
│   └── confusion_matrix.png          # 混淆矩阵
├── checkpoint/
│   ├── latest.pt                     # 最新模型
│   └── best.pt                       # 验证集最佳模型
├── tensorboard/                      # TensorBoard events
└── results/results.csv               # 实验结果指标
```

### 结果汇总

批量实验完成后：

```bash
# 汇总已有实验结果
python scripts/run_experiments.py --summarize-only

# 导出到 CSV
python scripts/run_experiments.py --summarize-only --export all_results.csv
```

### 查看 LLM 规则

```bash
python scripts/view_llm_rules.py
```

读取 `outputs/<exp_name>/run_*/llm_rules.json` 展示 MPFC 内置 LLM 生成的 IF-THEN 规则。
