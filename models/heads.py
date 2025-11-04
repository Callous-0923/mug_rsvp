# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TriHead(nn.Module):
    def __init__(self, dim_in, n_cls=3):
        super().__init__()
        self.fc = nn.Linear(dim_in, n_cls)
    def forward(self, f): return self.fc(f)

class BinHead(nn.Module):
    def __init__(self, dim_in, n_cls=2):
        super().__init__()
        self.fc = nn.Linear(dim_in, n_cls)
    def forward(self, f): return self.fc(f)

def tri_to_bin_probs(p_tri):
    """
    将三分类概率映射为二分类分布：non-target -> 0；target1/2 -> 1
    p_tri: (B,3) -> p_bin:(B,2)
    """
    p_bin = torch.zeros(p_tri.size(0), 2, device=p_tri.device, dtype=p_tri.dtype)
    p_bin[:, 0] = p_tri[:, 0]
    p_bin[:, 1] = p_tri[:, 1] + p_tri[:, 2]
    return p_bin

def hierarchical_self_distill(logits_tri, logits_bin):
    """
    与 main.py 的 self_distillation() 等价：三类->二类后，做对称 KL 作为层级自蒸馏【:contentReference[oaicite:11]{index=11}】。
    """
    p_tri = F.softmax(logits_tri, dim=-1)
    p_bin = F.softmax(logits_bin, dim=-1)
    p_tri2 = tri_to_bin_probs(p_tri)
    loss = 0.5*F.kl_div((p_tri2+1e-8).log(), p_bin, reduction='batchmean') + \
           0.5*F.kl_div((p_bin+1e-8).log(), p_tri2, reduction='batchmean')
    return loss
