# -*-coding: utf-8 -*-

import torch
from torch import Tensor

class Accuracy:
    """
    Top-1 classification accuracy.

    YAML:
        metrics:
          - _target_: src.metrics.Accuracy
            name: accuracy
    """
    def __init__(self, name: str = "accuracy"):
        self.name = name

    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            return (preds == labels).float().mean().item() * 100


class MacroF1:
    """
    Macro-averaged F1 score (requires sklearn).

    YAML:
        metrics:
          - _target_: src.metrics.MacroF1
            name: macro_f1
    """
    def __init__(self, name: str = "macro_f1", num_classes: int = 5):
        self.name = name
        self.num_classes = num_classes

    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        from sklearn.metrics import f1_score

        preds = logits.argmax(dim=1).cpu().numpy()
        y = labels.cpu().numpy()
        return f1_score(y, preds, average="macro", zero_division=0) * 100