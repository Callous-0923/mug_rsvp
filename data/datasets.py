import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import loadmat
import os
#PyTorch Dataset：EEG+EM 读取 & 拼接

class EEGEyeRSVPDataset(Dataset):
    """
    - EEG: 形状 (N, C_eeg, T)；- EM: (N, C_eye, T)
    - 按通道维拼接 -> (N, C_eeg + C_eye, T)，再扩成 (N, 1, C, T)
    """
    def __init__(self, data_root, task, subject, blocks, resample_hz=128,
                 eeg_chans=64, eye_chans=6, transforms=None):
        self.samples, self.labels, self.block_ids = self._load_all(
            data_root, task, subject, blocks, resample_hz, eeg_chans, eye_chans
        )
        self.transforms = transforms

    def _load_all(self, root, task, subject, blocks, resample_hz, eeg_chans, eye_chans):
        X, y, blk_ids, idx = None, None, None, 0
        for blk in blocks:
            # === 与你的 load_data_sd() 一致（路径/重采样/通道拼接）【:contentReference[oaicite:8]{index=8}】===
            eeg_path = os.path.join(root, task, "EEG", f"{subject}_{blk}.npz")
            print(eeg_path)
            eeg = np.load(eeg_path)
            data_eeg, label_eeg = eeg["data"], eeg["label"]
            data_eeg = signal.resample(data_eeg, resample_hz, axis=-1)
            label_eeg = np.squeeze(label_eeg)  # -> (N,)

            eye_path = os.path.join(root, task, "EM", f"{subject}_{blk}.mat")
            print(eye_path)
            em = loadmat(eye_path)
            data_eye = em["data"]   # (N, T, C_eye_raw) or (N, C_eye_raw, T) 视你的文件而定
            data_eye = np.transpose(data_eye, (0, 2, 1))    # 对齐为 (N, C, T)【:contentReference[oaicite:9]{index=9}】
            data_eye = data_eye[:, :eye_chans, :]
            data_eye = signal.resample(data_eye, resample_hz, axis=-1)
            label_eye = np.squeeze(em["label"])
            assert np.array_equal(label_eeg, label_eye), "EEG/EM标签不一致"

            data = np.concatenate([data_eeg, data_eye], axis=-2)  # (N, C_eeg+C_eye, T)
            X = data if X is None else np.concatenate([X, data], axis=0)
            y = label_eeg if y is None else np.concatenate([y, label_eeg], axis=0)
            blk_ids = np.ones(data.shape[0]) * idx if blk_ids is None else np.concatenate([blk_ids, np.ones(data.shape[0]) * idx])
            idx += 1

        # 标准化（trial-wise）
        mean = X.mean(-1, keepdims=True)
        std  = X.std(-1, keepdims=True) + 1e-6
        X = (X - mean) / std
        X = X[:, None, :, :]  # -> (N, 1, C, T)，与你训练脚本一致【:contentReference[oaicite:10]{index=10}】
        # 生成二分类标签（T vs Non-T），与 main.py 对齐【:contentReference[oaicite:11]{index=11}】
        y_bin = (y != 0).astype(np.int64)
        y = np.stack([y, y_bin], axis=-1)  # (N, 2)
        return X.astype(np.float32), y.astype(np.int64), blk_ids.astype(np.int64)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x = self.samples[idx]
        y_tri, y_bin = self.labels[idx, 0], self.labels[idx, 1]
        return torch.from_numpy(x), torch.tensor([y_tri, y_bin], dtype=torch.long)
