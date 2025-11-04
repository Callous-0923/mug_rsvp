# -*- coding: utf-8 -*-
from typing import List, Tuple
import torch

def default_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    将 [(x_i, y_i=[tri,bin]), ...] 拼为 batch。
    x: (1,C,T) -> (B,1,C,T); y: (2,) -> (B,2)
    """
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0).float()
    ys = torch.stack(ys, dim=0).long()
    return xs, ys
