# -*- coding: utf-8 -*-
import numpy as np
import torch

def confusion(y_true, y_pred, n_classes=None):
    """
    返回混淆矩阵 (n_classes, n_classes), 行=真类, 列=预测类
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def balanced_acc(y_true, y_pred, n_classes=None, eps=1e-12):
    """
    Balanced Accuracy = (1/K) * sum_c (TP_c / P_c)
    """
    cm = confusion(y_true, y_pred, n_classes)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(cm) / (cm.sum(axis=1) + eps)
    per_class[np.isnan(per_class)] = 0.0
    return float(per_class.mean())

def tensor2np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
