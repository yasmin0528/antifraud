"""
VTA (Ventral Tegmental Area) —— 多巴胺奖励预测误差（RPE）信号模块。

类脑机制：
VTA 多巴胺神经元编码奖励预测误差（Reward Prediction Error），
作为神经调质信号影响前脑多个区域（CA1、CA3、mPFC）的计算过程。

重构为真正的调控信号（而非静态损失加权）：
1. RPE = |y - prob| ：当前预测与真实标签的偏差
2. da_signal = 1 + beta * RPE ：转换为多巴胺调控信号
3. da_signal 作为前向控制信号影响：
   - CA3：增强/减弱记忆读取强度（逐样本调制）
   - MPFC：调节注意力锐利度（规则导向注意力的强度）
   - Loss：Focal loss 的 gamma 参数随 RPE 自适应调整
4. RPE 越高 → 学习信号越强 → 模型更关注困难样本
"""

from typing import Optional

import torch


def compute_da_signal(
    prob: torch.Tensor,
    y: torch.Tensor,
    rpe_beta: float = 1.5,
    momentum: float = 0.9,
    prev_da: torch.Tensor = None,
) -> torch.Tensor:
    """
    计算多巴胺 RPE 调控信号。

    这是一个纯前向的控制信号，不参与 loss 计算。
    高 RPE（预测错误） → 高 da_signal → 增强 CA3 记忆读取 + MPFC 注意力锐利度
    低 RPE（预测正确） → 低 da_signal → 保持正常计算

    Args:
        prob: [batch_size, 1] 当前预测概率
        y: [batch_size, 1] 真实标签
        rpe_beta: RPE 信号强度系数
        momentum: 时序平滑动量（0=不平滑，完全依赖当前 RPE）
        prev_da: [batch_size, 1] 上一步的 da_signal（None 则用当前值）

    Returns:
        da_signal: [batch_size, 1] 多巴胺调控信号（值域 [1.0, 1.0 + rpe_beta]）
    """
    y = y.float().view(-1, 1)
    prob = prob.view(-1, 1).clamp(1e-6, 1 - 1e-6)

    # RPE = |y - prob|，预测误差的绝对值
    delta = (y - prob).abs().detach()  # 停止梯度传播（调控信号不参与反向传播）

    # da_signal = 1 + beta * RPE
    da_current = 1.0 + rpe_beta * delta

    # 时序平滑（可选）
    if prev_da is not None and prev_da.size(0) == da_current.size(0):
        da_signal = momentum * prev_da + (1 - momentum) * da_current
    else:
        da_signal = da_current

    return da_signal


def compute_global_da_signal(
    prob: torch.Tensor,
    y: torch.Tensor,
    rpe_beta: float = 1.5,
    momentum: float = 0.9,
    prev_da: Optional[float] = None,
) -> float:
    """
    计算全局（epoch级）多巴胺 RPE 信号。

    对全量训练集节点取 RPE 均值，产生一个全局标量调制信号。
    这是 graph_node_trainer 场景下的默认模式：
    - 一个标量 da_signal 广播到所有节点，统一调控注意力锐利度
    - 当预测整体偏误时（如新 epoch 开始），全局增强学习信号

    Args:
        prob: [N] 当前预测概率（全量节点的预测值）
        y: [N] 真实标签
        rpe_beta: RPE 信号强度系数
        momentum: 时序平滑动量
        prev_da: 上一步 da_signal 的标量值（None 则用当前值）

    Returns:
        da_signal: float 标量（值域 [1.0, 1.0 + rpe_beta]）
    """
    prob = prob.detach().clamp(1e-6, 1 - 1e-6)
    y = y.detach().float()

    # 全局 RPE = 所有训练集节点的 |y - prob| 均值
    rpe = (y - prob).abs().mean().item()

    # 全局 DA 信号
    da_current = 1.0 + rpe_beta * rpe

    # 时序平滑
    if prev_da is not None:
        da_signal = momentum * prev_da + (1 - momentum) * da_current
    else:
        da_signal = da_current

    return da_signal
