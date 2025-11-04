# -*- coding: utf-8 -*-
# 滤波/重采样/标准化/伪迹处理

import numpy as np
from scipy.signal import butter, sosfiltfilt

def bandpass_sos(data, fs, lo=0.1, hi=30.0, order=4):
    """
    data: (N,C,T) 或 (C,T)
    返回与输入同形状的带通滤波结果。
    """
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    if data.ndim == 3:
        out = np.stack([sosfiltfilt(sos, d, axis=-1) for d in data], axis=0)
    else:
        out = sosfiltfilt(sos, data, axis=-1)
    return out

def zscore_trial(x, eps=1e-6):
    """
    trial-wise 标准化：x shape (C,T) -> (C,T)
    """
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True) + eps
    return (x - m) / s
