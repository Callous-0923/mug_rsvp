# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset

def numpy_to_loader(X, Y, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl
