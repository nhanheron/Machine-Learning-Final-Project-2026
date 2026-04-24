"""
DenseNet-121 wrapper for CheXpert multi-label classification.

Supports two output modes:
  - Binary (default): one sigmoid output per pathology  -> BCE loss
  - MultiClass:       three softmax outputs per pathology -> CE loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CheXpertModel(nn.Module):
    """
    Parameters
    ----------
    num_classes : int
        Number of pathologies (default 5 for the competition tasks).
    multiclass : bool
        If True, output 3 logits per pathology (neg / pos / uncertain)
        for the U-MultiClass strategy.
    pretrained : bool
        Use ImageNet-pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 5,
        multiclass: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.multiclass = multiclass

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features  # 1024

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if multiclass:
            # 3 logits (neg, pos, uncertain) per pathology
            self.classifier = nn.Linear(in_features, num_classes * 3)
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        feats = nn.functional.relu(feats, inplace=True)
        feats = self.pool(feats).flatten(1)
        logits = self.classifier(feats)

        if self.multiclass:
            # Reshape to (batch, num_classes, 3) for per-pathology softmax
            logits = logits.view(-1, self.num_classes, 3)

        return logits
