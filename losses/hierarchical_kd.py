# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def hierarchical_self_distill(logits_tri, logits_bin):
    """
    与 main.py 的 self_distillation() 一致：三类->二类，再做对称 KL【:contentReference[oaicite:12]{index=12}】。
    """
    p_tri = F.softmax(logits_tri, dim=-1)
    p_bin = F.softmax(logits_bin, dim=-1)
    p_tri2 = torch.zeros_like(p_bin)
    p_tri2[:, 0] = p_tri[:, 0]
    p_tri2[:, 1] = p_tri[:, 1] + p_tri[:, 2]
    loss = 0.5*F.kl_div((p_tri2+1e-8).log(), p_bin, reduction='batchmean') + \
           0.5*F.kl_div((p_bin+1e-8).log(), p_tri2, reduction='batchmean')
    return loss
