# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def contribution_weights(y_true, logits_eeg, logits_eye):
    """
    与 main.py 的 contribution() 等价：使用单模态真类概率计算模态“贡献权”【:contentReference[oaicite:10]{index=10}】。
    y_true: (B,) tri-class; logits_*: (B,3)
    return: (B,2) -> [w_eeg, w_eye]
    """
    p_eeg = F.softmax(logits_eeg, dim=-1)
    p_eye = F.softmax(logits_eye, dim=-1)
    onehot = F.one_hot(y_true, num_classes=3).float()
    pe = (p_eeg * onehot).sum(-1)
    py = (p_eye * onehot).sum(-1)
    denom = (pe + py).clamp_min(1e-6)
    w_eeg = pe / denom
    w_eye = py / denom
    return torch.stack([w_eeg, w_eye], dim=-1)

def mug_weights(w_contrib, u_eeg, u_eye):
    """
    MUG权重 = 贡献 × 置信；置信 = 1/(1+不确定性)
    u_eeg/u_eye: (B,)
    """
    conf = torch.stack([1/(1+u_eeg), 1/(1+u_eye)], dim=-1).clamp_(0., 1.)
    return (w_contrib * conf)

class GatingFusion(nn.Module):
    """
    简单门控层：concat(eeg_feat, eye_feat) -> 2维 Softmax 权重
    """
    def __init__(self, dim_in):
        super().__init__()
        self.fc = nn.Linear(dim_in, 2)

    def forward(self, f_eeg, f_eye):
        h = torch.cat([f_eeg, f_eye], dim=-1)
        gates = F.softmax(self.fc(h), dim=-1)  # (B,2)
        return gates, h
