# -*- coding: utf-8 -*-
import torch.nn as nn

class GateAlignmentLoss(nn.Module):
    """
    门控监督：L1(gates, target_weights)。
    target_weights 可取 “贡献权” 或 “MUG权重(贡献×置信)”。
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, gates, target_weights):
        return self.l1(gates, target_weights)
