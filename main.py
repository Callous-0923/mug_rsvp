import os, time
from collections import defaultdict

import torch, numpy as np
from utils.config import load_config
from utils.seed import set_seed
from data.datasets import EEGEyeRSVPDataset
from torch.utils.data import DataLoader, TensorDataset
from models.mug_rsvp_net import MUGRSVPNet
from engine.train import train_one_epoch
from engine.validate import validate
from engine.test import test_and_export
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

def main():
    cfg = load_config()
    print("配置参数： ",cfg)

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg["save_dir"]; os.makedirs(save_dir, exist_ok=True)
    print("保存路径： ",save_dir)
    subject_acc = defaultdict(lambda: defaultdict(list))
    task_acc = defaultdict(list)

    for task in cfg["tasks"]:
        for subject in cfg["subjects"]:
            # 加载完整数据（含所有块）
            dataset = EEGEyeRSVPDataset(cfg["data_root"], task, subject, cfg["blocks"],
                                         resample_hz=cfg["resample_hz"],
                                         eeg_chans=cfg["eeg_chans"], eye_chans=cfg["eye_chans"])
            X = dataset.samples; Y = dataset.labels  # (N,1,C,T), (N,2)
            print("数据形状： X.shape",X.shape, "Y.shape",Y.shape)
            block_ids = dataset.block_ids

            for block in cfg["blocks"]:
                # 留出某一块做测试，其余作为训练+验证（与你现有流程相同）【:contentReference[oaicite:24]{index=24}】
                train_mask = (block_ids != (block-1))
                test_mask  = ~train_mask
                X_train, Y_train = X[train_mask], Y[train_mask]
                X_test, Y_test   = X[test_mask], Y[test_mask]
                print("============已完成训练集测试集划分=============")
                # 类均衡（可使用 WeightedRandomSampler 或按你现有“最小类裁剪”策略）【:contentReference[oaicite:25]{index=25}】
                y_tri = Y_train[:,0]
                classes = np.unique(y_tri)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_tri)
                w_tri = torch.tensor(class_weights, dtype=torch.float32, device=device)
                w_bin = torch.tensor(compute_class_weight('balanced', classes=np.unique(Y_train[:,1]), y=Y_train[:,1]),
                                     dtype=torch.float32, device=device)
                print("============已完成类均衡=============")

                # K 折交叉验证
                print("============开始K折交叉验证 START K-FOLD CROSS VALIDATION=============")
                skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=42)
                block_fold_accs = []
                for fold, (idx_tr, idx_val) in enumerate(skf.split(X_train, y_tri)):
                    tr_ds = TensorDataset(torch.from_numpy(X_train[idx_tr]), torch.from_numpy(Y_train[idx_tr]))
                    va_ds = TensorDataset(torch.from_numpy(X_train[idx_val]), torch.from_numpy(Y_train[idx_val]))
                    te_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
                    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
                    va_dl = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False)
                    te_dl = DataLoader(te_ds, batch_size=cfg["batch_size"], shuffle=False)
                    print("============数据加载完成=============")

                    model = MUGRSVPNet(dropout=cfg["dropout"]).to(device)
                    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                             betas=tuple(cfg["betas"]))
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5)  #【:contentReference[oaicite:26]{index=26}】

                    best_val, best_path = 1e9, os.path.join(save_dir, f"{task}_{subject}_blk{block}_fold{fold}.pth")
                    no_improve, patience = 0, cfg["patience"]
                    print("============模型导入完成=============")

                    for epoch in range(cfg["epochs"]):
                        epoch_start = time.time()
                        m = train_one_epoch(model, tr_dl, optim, device, cfg,
                                            class_weights_tri=w_tri, class_weights_bin=w_bin)
                        print("============已训练单次epoch=============")
                        val = validate(model, va_dl, device, class_weights_tri=w_tri)
                        epoch_time = time.time() - epoch_start
                        print(epoch_time)
                        scheduler.step(val["val_ba"])

                        if (epoch+1) % 10 == 0:
                            print(f"[{task}/{subject}/blk{block}/fold{fold}] "
                                  f"ep{epoch+1} train_loss={m['loss']:.4f} "
                                  f"val_loss={val['val_loss']:.4f} val_ba={val['val_ba']:.4f} "
                                  f"time={epoch_time:.2f}s")

                        if val["val_loss"] < best_val:
                            best_val, no_improve = val["val_loss"], 0
                            torch.save(model.state_dict(), best_path)
                        else:
                            no_improve += 1
                        if no_improve > patience: break

                    # 测试并导出
                    print("============开始测试并导出=============")
                    model.load_state_dict(torch.load(best_path, map_location=device))
                    acc, ba, cm = test_and_export(model, te_dl, device, save_dir,
                                                  tag=f"{task}_{subject}_blk{block}_fold{fold}")
                    block_fold_accs.append(acc)
                if block_fold_accs:
                    block_mean_acc = float(np.mean(block_fold_accs))
                    subject_acc[task][subject].append(block_mean_acc)
                    task_acc[task].append(block_mean_acc)
                    subject_mean_acc = float(np.mean(subject_acc[task][subject]))
                    task_mean_acc = float(np.mean(task_acc[task]))
                    print(f"[{task}/{subject}] block {block} finished: block_acc={block_mean_acc:.4f} "
                          f"subject_acc={subject_mean_acc:.4f} task_acc={task_mean_acc:.4f}")

if __name__ == "__main__":
    main()



