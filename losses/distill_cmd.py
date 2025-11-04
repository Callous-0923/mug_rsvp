# -*- coding: utf-8 -*-
"""
跨模态蒸馏（可选）：适配 CMD 四阶段中的常见损失：
- 特征蒸馏（MSE/Huber）
- 响应蒸馏（KL）
- 掩码重建（CAMAE思路；token级重建）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def kd_logits_kl(student_logits, teacher_logits, T=3.0):
    ps = F.log_softmax(student_logits / T, dim=-1)
    pt = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(ps, pt, reduction='batchmean') * (T*T)

def kd_feat_mse(student_feat, teacher_feat):
    return F.mse_loss(student_feat, teacher_feat)

class MaskedReconstructionLoss(nn.Module):
    """
    简化版 CAMAE：随机mask部分特征，学生重建被mask的部分（从教师特征蒸馏）
    """
    def __init__(self, mask_ratio=0.25):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.l1 = nn.L1Loss()

    def forward(self, student_feat, teacher_feat):
        B, D = student_feat.shape
        m = int(D * self.mask_ratio)
        idx = torch.randperm(D, device=student_feat.device)[:m]
        s_masked = student_feat[:, idx]
        t_masked = teacher_feat[:, idx].detach()
        return self.l1(s_masked, t_masked)
