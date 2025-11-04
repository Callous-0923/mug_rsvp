# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        return self.ce(logits, targets)

class FocalLoss(nn.Module):
    """
    经典 Focal Loss (可用于类别不平衡)。
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, targets):
        logpt = F.log_softmax(logits, dim=-1)
        pt = torch.exp(logpt)
        logpt = (logpt * F.one_hot(targets, num_classes=logits.size(-1))).sum(-1)
        pt = (pt * F.one_hot(targets, num_classes=logits.size(-1))).sum(-1)
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
