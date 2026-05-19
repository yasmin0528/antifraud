"""
VTA (Ventral Tegmental Area) 损失函数 —— 多巴胺奖励加权损失。

类脑机制：
- 正类加权（pos_weight）：模拟多巴胺对罕见正类的高敏感性
- Focal loss（focal_gamma）：降低易分类样本的权重，聚焦难样本
- RPE（Reward Prediction Error）：预测误差越大，学习信号越强
"""

import torch
import torch.nn.functional as F


def vta_weighted_loss(
    logit: torch.Tensor,
    y: torch.Tensor,
    prob: torch.Tensor = None,
    pos_weight: float = 5.0,
    focal_gamma: float = 2.0,
    rpe_beta: float = 1.5,
) -> torch.Tensor:
    """
    VTA 加权损失函数。

    loss = mean( BCE * Focal * DA * PW )

    其中:
    - BCE: 二值交叉熵
    - Focal = (1 - pt)^gamma: 聚焦于难分类样本
    - DA = 1 + beta * |y - prob|: 奖励预测误差调制
    - PW: 正类权重

    Args:
        logit: [batch_size, 1] 未归一化 logit
        y: [batch_size, 1] 真实标签
        prob: [batch_size, 1] 预测概率（如为 None 则从 logit 计算）
        pos_weight: 正类权重
        focal_gamma: Focal loss gamma
        rpe_beta: 奖励预测误差系数

    Returns:
        loss: 标量损失值
    """
    y = y.float().view(-1, 1)

    if prob is None:
        prob = torch.sigmoid(logit)
    prob = prob.clamp(1e-6, 1 - 1e-6)

    # BCE
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")

    # Focal modulation
    pt = torch.where(y == 1, prob, 1 - prob)
    focal = (1 - pt) ** focal_gamma

    # Reward Prediction Error (dopamine-like)
    delta = (y - prob).abs().detach()
    da = 1.0 + rpe_beta * delta

    # Positive weight
    pw = torch.where(
        y == 1,
        torch.tensor(pos_weight, device=y.device),
        torch.tensor(1.0, device=y.device),
    )

    return (bce * focal * da * pw).mean()
