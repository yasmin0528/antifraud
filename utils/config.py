"""
配置文件加载与管理。

支持：
- YAML 配置文件加载
- 命令行参数覆盖（via argparse）
- 多配置合并（默认 + 实验特定）
- 类型安全的 dataclass 访问
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import yaml


# ============================================================
# 类型安全的配置类（对应 default.yaml）
# ============================================================


@dataclass
class ExperimentConfig:
    name: str = "aml_baseline"
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "auto"
    mode: str = "train"


@dataclass
class DataConfig:
    dataset: str = "aml"
    data_path: str = "dataset.csv"
    preprocessed_path: str = "preprocessed_data.pt"
    regenerate: bool = False
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    use_smote: bool = False
    smote_ratio: float = 1.0
    batch_size: int = 32
    num_workers: int = 0
    window_size: int = 10                     # 滑动窗口大小（cryptopia 数据集使用）


@dataclass
class CA1Config:
    feature_dim: int = 3
    hidden_dim: int = 128


@dataclass
class CA3Config:
    emb_dim: int = 128
    num_groups: int = 16
    memory_momentum: float = 0.9
    memory_mode: str = "explicit_group"
    update_mode: str = "ema_group_proto"


@dataclass
class MPFCConfig:
    gnn_layers: int = 2
    gnn_heads: int = 4
    input_dim: Optional[int] = None


@dataclass
class LLMConfig:
    model_name: str = "qwen"
    use_api: bool = True
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    rule_update_frequency: int = 100
    max_rules: int = 20


@dataclass
class ModelConfig:
    name: str = "MPFC"
    hidden_dim: int = 128
    dim_type: int = 16
    dropout: float = 0.2
    edge_attr_dim: int = 3
    task_dim: int = 128
    ca1: CA1Config = field(default_factory=CA1Config)
    ca3: CA3Config = field(default_factory=CA3Config)
    mpfc: MPFCConfig = field(default_factory=MPFCConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 0.0001
    pos_weight: float = 5.0
    focal_gamma: float = 2.0
    rpe_beta: float = 1.5
    patience: int = 5
    grad_clip: float = 1.0
    log_interval: int = 200


@dataclass
class EvalConfig:
    enable_alert_metrics: bool = True
    enable_subgraph_metrics: bool = True
    alert_agg: str = "max"
    hit_k: int = 10
    eval_da_mode: str = "fixed_identity"


@dataclass
class ExplainConfig:
    export_attention: bool = True
    export_rule_trace: bool = True


@dataclass
class VTAConfig:
    mode: str = "batch_scalar"
    target_modules: List[str] = field(
        default_factory=lambda: ["ca1", "ca3", "mpfc_rule", "mpfc_gate", "mpfc_output"]
    )


@dataclass
class AblationConfig:
    enabled: bool = False
    remove_modules: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)


@dataclass
class SweepConfig:
    enabled: bool = False
    method: str = "grid"
    params: Dict[str, Optional[List[Any]]] = field(default_factory=lambda: {
        "batch_size": None,
        "lr": None,
        "hidden_dim": None,
        "dropout": None,
        "pos_weight": None,
        "focal_gamma": None,
        "gnn_layers": None,
        "gnn_heads": None,
    })


@dataclass
class MultiSeedConfig:
    enabled: bool = False
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1111])


@dataclass
class VisualizationConfig:
    enabled: bool = True
    save_figures: bool = True
    plot_loss: bool = True
    plot_roc: bool = True
    plot_tsne: bool = False
    plot_attention: bool = False
    plot_model_graph: bool = False


@dataclass
class VTAVariantConfig:
    name: str = ""
    focal_gamma: float = 2.0
    rpe_beta: float = 1.5
    pos_weight: float = 5.0


@dataclass
class VTADecompConfig:
    enabled: bool = False
    variants: List[Dict] = field(default_factory=lambda: [
        {"name": "full", "focal_gamma": 2.0, "rpe_beta": 1.5, "pos_weight": 5.0},
        {"name": "wo_focal", "focal_gamma": 0.0, "rpe_beta": 1.5, "pos_weight": 5.0},
        {"name": "wo_rpe", "focal_gamma": 2.0, "rpe_beta": 0.0, "pos_weight": 5.0},
        {"name": "wo_pw", "focal_gamma": 2.0, "rpe_beta": 1.5, "pos_weight": 1.0},
        {"name": "bce_only", "focal_gamma": 0.0, "rpe_beta": 0.0, "pos_weight": 1.0},
    ])


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    vta: VTAConfig = field(default_factory=VTAConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    multi_seed: MultiSeedConfig = field(default_factory=MultiSeedConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    vta_decomp: VTADecompConfig = field(default_factory=VTADecompConfig)

    def __post_init__(self):
        # 将嵌套 dict 转为 dataclass
        if isinstance(self.experiment, dict):
            self.experiment = ExperimentConfig(**self.experiment)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            model_dict = dict(self.model)
            for key in ("ca1", "ca3", "mpfc", "llm"):
                if key in model_dict and isinstance(model_dict[key], dict):
                    model_dict[key] = _SUBCONFIG_MAP[key](**model_dict[key])
            self.model = ModelConfig(**model_dict)
        if isinstance(self.train, dict):
            self.train = TrainConfig(**self.train)
        if isinstance(self.eval, dict):
            self.eval = EvalConfig(**self.eval)
        if isinstance(self.explain, dict):
            self.explain = ExplainConfig(**self.explain)
        if isinstance(self.vta, dict):
            self.vta = VTAConfig(**self.vta)
        if isinstance(self.ablation, dict):
            self.ablation = AblationConfig(**self.ablation)
        if isinstance(self.sweep, dict):
            self.sweep = SweepConfig(**self.sweep)
        if isinstance(self.multi_seed, dict):
            self.multi_seed = MultiSeedConfig(**self.multi_seed)
        if isinstance(self.visualization, dict):
            self.visualization = VisualizationConfig(**self.visualization)
        if isinstance(self.vta_decomp, dict):
            self.vta_decomp = VTADecompConfig(**self.vta_decomp)

    def to_dict(self) -> Dict:
        return asdict(self)


_SUBCONFIG_MAP = {
    "ca1": CA1Config,
    "ca3": CA3Config,
    "mpfc": MPFCConfig,
    "llm": LLMConfig,
}


# ============================================================
# 加载与合并
# ============================================================


def load_config(path: str) -> Config:
    """从 YAML 文件加载配置。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(**raw)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Config):
    """验证配置的有效性。

    LLM 是 mPFC 模块的内置组件。当使用 LLM 时（非 wo_llm 消融模式），
    需要配置 API URL 或模型名称。
    """
    is_wo_llm = "wo_llm" in cfg.ablation.remove_modules
    if not is_wo_llm:
        if not cfg.model.llm.api_url and not cfg.model.llm.model_name:
            raise ValueError(
                "LLM 是 mPFC 模块的内置组件，需要配置 model.llm.api_url（API 地址）"
                "或 model.llm.model_name（本地模型名称）。\n"
                "请检查配置文件中的 llm 节，或在命令行通过 "
                "--llm_api_url / --llm_model_name 指定。\n"
                "如需运行 wo_llm 消融实验（去除 LLM），请设置 "
                "ablation.remove_modules 包含 'wo_llm'。"
            )


def merge_config(base: Config, override: Dict) -> Config:
    """将命令行参数或另一个 YAML 的配置合并到基础配置上。"""
    base_dict = asdict(base)

    def _deep_merge(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if k in d and isinstance(d[k], dict) and isinstance(v, dict):
                _deep_merge(d[k], v)
            else:
                d[k] = v
        return d

    merged = _deep_merge(base_dict, override)
    cfg = Config(**merged)
    _validate_config(cfg)
    return cfg
