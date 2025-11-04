# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeConvBackbone(nn.Module):
    """
    轻量 EM/眼动分支：跨通道卷积 + 时域卷积 -> 128维
    参考您 EM_feature 的“先跨通道卷积，再展平”的做法【:contentReference[oaicite:9]{index=9}】。
    """
    def __init__(self, eye_chans=6, base_filters=32, dropout=0.2):
        super().__init__()
        self.conv_c = nn.Conv2d(1, base_filters, (eye_chans, 1), stride=(eye_chans, 1), bias=False)
        self.bn_c = nn.BatchNorm2d(base_filters)
        self.conv_t = nn.Conv2d(base_filters, base_filters*2, (1, 8), stride=(1, 8), bias=False)
        self.bn_t = nn.BatchNorm2d(base_filters*2)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.proj = nn.Linear(base_filters*2, 128)

    def forward(self, x):
        # x: (B,1,C_eye,T)
        z = F.leaky_relu(self.bn_c(self.conv_c(x)), inplace=True)
        z = self.dropout(z)
        z = F.leaky_relu(self.bn_t(self.conv_t(z)), inplace=True)
        z = self.dropout(z)
        z = self.pool(z).squeeze(-1).squeeze(-1)  # (B, base_filters*2)
        z = self.proj(z)                          # (B, 128)
        return z
