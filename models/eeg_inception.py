# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGInceptionBackbone(nn.Module):
    """
    多尺度卷积 + 深度可分空间卷积（跨通道）-> 384维特征
    输入:  x_eeg (B,1,C_eeg,T)；输出: feat(B,384), mid(B,mid_dim)
    设计动机与多尺度结构参考您现有 EEG_feature（分支堆叠+下采样）【:contentReference[oaicite:7]{index=7}】。
    """
    def __init__(self, eeg_chans=64, base_filters=64, kernels=(64,32,16,8), dropout=0.5):
        super().__init__()
        self.eeg_chans = eeg_chans
        self.dropout = nn.Dropout(dropout)

        # branch: 时域卷积 -> 深度可分空间卷积(跨通道) -> 1x1
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(nn.Sequential(
                nn.ConstantPad2d((0,0,k//2, k//2 - 1 if k%2==0 else k//2), 0.0),
                nn.Conv2d(1, base_filters, (k, 1), bias=False),
                nn.BatchNorm2d(base_filters),
                nn.ELU(inplace=True),
                nn.Dropout(p=dropout),

                nn.Conv2d(base_filters, base_filters, (1, eeg_chans), groups=base_filters, bias=False),
                nn.Conv2d(base_filters, base_filters, (1, 1), bias=False),
                nn.BatchNorm2d(base_filters),
                nn.ELU(inplace=True),
                nn.Dropout(p=dropout),
            ))

        # second stage: 融合后再做短核卷积（粗细粒度融合）
        self.fuse_blocks = nn.ModuleList()
        for k in [s//4 if isinstance(s,int) else 8 for s in kernels]:
            kk = int(max(3, k))
            self.fuse_blocks.append(nn.Sequential(
                nn.ConstantPad2d((0,0,kk//2, kk//2 - 1 if kk%2==0 else kk//2), 0.0),
                nn.Conv2d(len(kernels)*base_filters, base_filters//2, (kk,1), bias=False),
                nn.BatchNorm2d(base_filters//2),
                nn.ELU(inplace=True),
                nn.Dropout(p=dropout),
            ))

        # 输出维度：len(kernels)*(base_filters//2) = 4*(32)=128 -> 再线性扩展到 384
        self.out_pool = nn.AdaptiveAvgPool2d((1,1))
        self.proj_mid = nn.Linear(len(kernels)*(base_filters//2), 192)
        self.proj_out = nn.Linear(192, 384)

    def forward(self, x):
        # x: (B,1,C_eeg,T) -> 与您 model 里先 permute 再卷积的习惯对齐【:contentReference[oaicite:8]{index=8}】
        # 这里沿用 (B,1,T,C) vs (B,1,C,T) 的实现差别不大，保持 (B,1,C,T) 即可。
        b1_outs = [b(x) for b in self.branches]          # list of (B,F,T',1)
        b1_out = torch.cat(b1_outs, dim=1)               # (B, F*len(k), T',1)
        b1_out = F.avg_pool2d(b1_out, (4,1))

        b2_outs = [blk(b1_out) for blk in self.fuse_blocks]
        b2_out = torch.cat(b2_outs, dim=1)               # (B, len(k)*(F//2), T'',1)
        b2_out = F.avg_pool2d(b2_out, (2,1))

        # 池化+投影
        pooled = self.out_pool(b2_out).squeeze(-1).squeeze(-1)  # (B, len(k)*(F//2))
        mid = self.proj_mid(pooled)                              # (B, 192)
        mid = F.elu(self.dropout(mid), inplace=True)
        feat = self.proj_out(mid)                                # (B, 384)
        return feat, b1_out.flatten(1)   # 返回早期特征用于蒸馏/可视化
