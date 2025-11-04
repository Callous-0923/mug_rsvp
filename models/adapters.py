# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class SubjectAdapter(nn.Module):
    """
    被试特定 Adapter：轻量瓶颈层，插在特征后 (可选)。
    """
    def __init__(self, dim, bottleneck=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim),
        )
    def forward(self, f):
        return f + self.net(f)
