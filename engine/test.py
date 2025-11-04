import numpy as np, scipy.io as sio, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from .metrics import balanced_acc, confusion

@torch.no_grad()
def test_and_export(model, dl, device, save_dir, tag="test"):
    model.eval()
    probs_tri, probs_eeg, probs_eye, probs_bin, labels_tri, labels_bin = [], [], [], [], [], []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        y_tri = y[:, 0].cpu().numpy()
        y_bin = y[:, 1].cpu().numpy()
        tri_fused, tri_eeg, tri_eye, bin_logits, gates = model(x)
        probs_tri.append(torch.softmax(tri_fused, dim=-1).cpu().numpy())
        probs_eeg.append(torch.softmax(tri_eeg, dim=-1).cpu().numpy())
        probs_eye.append(torch.softmax(tri_eye, dim=-1).cpu().numpy())
        probs_bin.append(torch.softmax(bin_logits, dim=-1).cpu().numpy())
        labels_tri.append(y_tri); labels_bin.append(y_bin)
    P = np.vstack(probs_tri); Pe = np.vstack(probs_eeg); Py = np.vstack(probs_eye); Pb = np.vstack(probs_bin)
    Y = np.concatenate(labels_tri); Yb = np.concatenate(labels_bin)
    acc = float((P.argmax(-1)==Y).mean()); ba = balanced_acc(Y, P.argmax(-1))
    cm  = confusion(Y, P.argmax(-1))
    # 保存 .mat 与 .json
    os.makedirs(save_dir, exist_ok=True)
    sio.savemat(os.path.join(save_dir, f"{tag}_scores.mat"), {"probs":P,"probs_eeg":Pe,"probs_eye":Py,"probs_bin":Pb,"y":Y,"y_bin":Yb})
    with open(os.path.join(save_dir, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"acc":acc, "ba":ba, "conf_mat":cm.tolist()}, f, ensure_ascii=False, indent=2)
    return acc, ba, cm
