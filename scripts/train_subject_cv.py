# -*- coding: utf-8 -*-
"""
跨任务/被试/块的交叉验证训练脚本：
- 与您 main.py 的训练流程一致：留一块为测试、其余块做 K 折 CV、ReduceLROnPlateau 调度、保存最优再测试导出。
- 数据读取与 trial-wise 标准化沿用 load_data_sd 思路（已封装到 EEGEyeRSVPDataset）。:contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from data.datasets import EEGEyeRSVPDataset
from models.mug_rsvp_net import MUGRSVPNet
from engine.train import train_one_epoch
from engine.validate import validate
from engine.test import test_and_export
from engine.checkpoint import BestKeeper
from utils.seed import set_seed

def run_train(cfg):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg["save_dir"]

    for task in cfg["tasks"]:
        for subject in cfg["subjects"]:
            dataset = EEGEyeRSVPDataset(cfg["data_root"], task, subject, cfg["blocks"],
                                         resample_hz=cfg["resample_hz"],
                                         eeg_chans=cfg["eeg_chans"], eye_chans=cfg["eye_chans"])
            X = dataset.samples  # (N,1,C,T)
            Y = dataset.labels   # (N,2)
            block_ids = dataset.block_ids

            for block in cfg["blocks"]:
                mask_tr = (block_ids != (block-1))
                X_tr, Y_tr = X[mask_tr], Y[mask_tr]
                X_te, Y_te = X[~mask_tr], Y[~mask_tr]

                # 类平衡权重（与 main.py 对齐）:contentReference[oaicite:6]{index=6}
                w_tri = compute_class_weight('balanced',
                                             classes=np.unique(Y_tr[:,0]),
                                             y=Y_tr[:,0])
                w_bin = compute_class_weight('balanced',
                                             classes=np.unique(Y_tr[:,1]),
                                             y=Y_tr[:,1])
                w_tri = torch.tensor(w_tri, dtype=torch.float32, device=device)
                w_bin = torch.tensor(w_bin, dtype=torch.float32, device=device)

                skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=42)
                for fold, (idx_tr, idx_val) in enumerate(skf.split(X_tr, Y_tr[:,0])):
                    tr_ds = TensorDataset(torch.from_numpy(X_tr[idx_tr]), torch.from_numpy(Y_tr[idx_tr]))
                    va_ds = TensorDataset(torch.from_numpy(X_tr[idx_val]), torch.from_numpy(Y_tr[idx_val]))
                    te_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(Y_te))

                    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
                    va_dl = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False)
                    te_dl = DataLoader(te_ds, batch_size=cfg["batch_size"], shuffle=False)

                    model = MUGRSVPNet(dropout=cfg["dropout"]).to(device)
                    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                             weight_decay=cfg["weight_decay"],
                                             betas=tuple(cfg["betas"]))
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max',
                                                                           factor=0.5, patience=5)

                    best_path = os.path.join(save_dir, f"{task}_{subject}_blk{block}_fold{fold}.pth")
                    best = BestKeeper(mode='min', out_path=best_path)

                    for epoch in range(cfg["epochs"]):
                        m = train_one_epoch(model, tr_dl, optim, device, cfg,
                                            class_weights_tri=w_tri, class_weights_bin=w_bin)
                        val = validate(model, va_dl, device, class_weights_tri=w_tri)
                        scheduler.step(val["val_ba"])

                        if (epoch + 1) % 10 == 0:
                            print(f"[{task}/{subject}/blk{block}/fold{fold}] "
                                  f"ep{epoch+1} loss={m['loss']:.4f} val_ba={val['val_ba']:.4f}")

                        best.step(val["val_loss"], model)

                    # 测试+导出
                    model.load_state_dict(torch.load(best_path, map_location=device))
                    tag = f"{task}_{subject}_blk{block}_fold{fold}"
                    acc, ba, cm = test_and_export(model, te_dl, device, save_dir, tag)

if __name__ == "__main__":
    # 若需要可从 YAML 读取，这里给一个最小默认 cfg
    cfg = dict(
        data_root="D:/files/datasets/eegem",
        tasks=["A","B","C"],
        subjects=[f"S{i+1}" for i in range(5)],
        blocks=[1,2,3,4,5],
        resample_hz=128, eeg_chans=64, eye_chans=6,
        epochs=100, batch_size=128,
        optimizer="adam", lr=1e-3, weight_decay=0.0, betas=[0.9,0.999],
        scheduler="reduce_on_plateau", patience=100, n_folds=5, device="cuda",
        dropout=0.5, mc_dropout_passes=10,
        lambda_cls_eeg_eye=0.3, lambda_hsd=1.0, lambda_gate=1.0,
        save_dir="./save/MUG_RSVP",
    )
    os.makedirs(cfg["save_dir"], exist_ok=True)
    run_train(cfg)
