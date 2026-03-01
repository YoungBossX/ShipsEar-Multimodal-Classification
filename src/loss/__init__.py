# -*-coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor

class WeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss with optional class-balancing weights.

    在 YAML 中配置:
        loss:
          _target_: src.loss.WeightedCrossEntropyLoss

    class_weights 由 train.py 在 instantiate 后通过
    loss_fn.weight = weights.to(device)  动态注入，
    因为权重需要先从 DataLoader 里统计出来，再迁移到正确的 device。
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", None)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )