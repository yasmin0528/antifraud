"""
随机种子固定 —— 保证实验可复现。
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    固定所有随机种子。

    Args:
        seed: 随机种子值
        deterministic: 是否启用 cuDNN 确定性算法
                       （会略微降低性能但保证完全可复现）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch >= 1.9
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["PYTHONHASHSEED"] = str(seed)
