# -*- coding: utf-8 -*-
import os
import yaml

_DEFAULT = dict(
    data_root="D:/files/datasets/eegem",
    tasks=["A","B","C"], subjects=[f"S{i+1}" for i in range(5)], blocks=[1,2,3,4,5],
    resample_hz=128, eeg_chans=64, eye_chans=6,
    epochs=100, batch_size=128, lr=1e-3, weight_decay=0.0, betas=[0.9,0.999],
    n_folds=5, dropout=0.5, mc_dropout_passes=10,patience=100,
    lambda_cls_eeg_eye=0.3, lambda_hsd=1.0, lambda_gate=1.0,
    save_dir="./save/MUG_RSVP",
)

def load_config(path: str = "D:/files/codes/mug_rsvp/configs/default.yaml"):
    """
    读取 YAML 配置；若为空则返回内置默认。
    """
    if path is None:
        return _DEFAULT.copy()
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = _DEFAULT.copy(); out.update(cfg or {})
    return out
