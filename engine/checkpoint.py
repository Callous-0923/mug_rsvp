# -*- coding: utf-8 -*-
import os
import torch

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, map_location=None, strict=True):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return model

class BestKeeper:
    """
    维护验证集最优（根据最小 val_loss 或最大 val_metric）。
    """
    def __init__(self, mode='min', best_init=None, out_path='best.pth'):
        assert mode in ['min', 'max']
        self.mode = mode
        self.best = best_init if best_init is not None else (1e9 if mode=='min' else -1e9)
        self.path = out_path

    def step(self, score, model):
        improved = (score < self.best) if self.mode=='min' else (score > self.best)
        if improved:
            self.best = score
            save_checkpoint(model, self.path)
        return improved
