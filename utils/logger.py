"""
日志系统 —— 支持控制台输出 + TensorBoard。
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    统一日志器。

    用法：
        logger = Logger("outputs/logs/exp1")
        logger.info("Training started")
        logger.scalar("train/loss", 0.5, step=10)
    """

    def __init__(
        self,
        log_dir: str,
        name: str = "aml",
        level: int = logging.INFO,
        console: bool = True,
        log_file: bool = True,
    ):
        self.log_dir = log_dir
        if log_file:
            os.makedirs(log_dir, exist_ok=True)

        # Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()

        # 文件 handler（仅在 log_file=True 时创建）
        if log_file:
            log_path = os.path.join(log_dir, f"{name}.log")
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(fh)

        # 控制台 handler
        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s | %(message)s",
                datefmt="%H:%M:%S",
            ))
            self._logger.addHandler(ch)

        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def scalar(self, tag: str, value: float, step: int):
        """记录标量到 TensorBoard。"""
        self.tb_writer.add_scalar(tag, value, step)

    def log_scalar(self, tag: str, value: float, step: int):
        """log_scalar 别名，兼容旧接口。"""
        self.scalar(tag, value, step)

    def scalars(self, tag: str, value_dict: dict, step: int):
        """记录多个标量。"""
        self.tb_writer.add_scalars(tag, value_dict, step)

    def figure(self, tag: str, figure, step: int):
        """记录 matplotlib 图表。"""
        self.tb_writer.add_figure(tag, figure, step)

    def log_figure(self, tag: str, figure_or_path, step: int = 0):
        """log_figure 别名，兼容旧接口。支持传入文件路径或 figure 对象。"""
        if isinstance(figure_or_path, str):
            import matplotlib.image as mpimg
            fig = mpimg.imread(figure_or_path)
            self.tb_writer.add_image(tag, fig, step, dataformats="HWC")
        else:
            self.figure(tag, figure_or_path, step)

    def histogram(self, tag: str, values, step: int):
        """记录直方图。"""
        self.tb_writer.add_histogram(tag, values, step)

    def close(self):
        self.tb_writer.close()
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)
