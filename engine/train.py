import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import balanced_acc
from .uncertainty import mc_dropout_uncertainty
from losses.hierarchical_kd import hierarchical_self_distill
from models.fusion_mug import contribution_weights


def train_one_epoch(model, dl, optimizer, device, cfg, class_weights_tri=None, class_weights_bin=None):
    model.train()
    ce_tri = nn.CrossEntropyLoss(weight=class_weights_tri).to(device)
    ce_bin = nn.CrossEntropyLoss(weight=class_weights_bin).to(device)
    l1 = nn.L1Loss().to(device)

    meter = {"loss": 0.0, "ba": 0.0}
    preds, gts = [], []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        y_tri = y[:, 0]
        y_bin = y[:, 1]
        optimizer.zero_grad()
        tri_fused, tri_eeg, tri_eye, bin_logits, gates = model(x)

        # Contribution weights derived from unimodal logits.
        W_contrib = contribution_weights(y_tri, tri_eeg, tri_eye)

        # Uncertainty-aware modulation via MC-dropout.
        with torch.no_grad():
            u_eeg, u_eye = mc_dropout_uncertainty(model, x, passes=cfg["mc_dropout_passes"])
            conf = torch.stack([1 / (1 + u_eeg), 1 / (1 + u_eye)], dim=-1)
        W_mug = W_contrib * conf

        loss_tri = ce_tri(tri_fused, y_tri) + cfg["lambda_cls_eeg_eye"] * (
            ce_tri(tri_eeg, y_tri) + ce_tri(tri_eye, y_tri)
        )
        loss_bin = ce_bin(bin_logits, y_bin)
        loss_hsd = hierarchical_self_distill(tri_fused, bin_logits) * cfg["lambda_hsd"]
        loss_gate = l1(gates, W_mug.detach()) * cfg["lambda_gate"]

        loss = loss_tri + loss_bin + loss_hsd + loss_gate
        loss.backward()
        optimizer.step()

        meter["loss"] += float(loss.item()) * x.size(0)
        preds.append(tri_fused.argmax(dim=-1).detach().cpu())
        gts.append(y_tri.detach().cpu())

    meter["loss"] /= len(dl.dataset)
    if preds:
        y_true = torch.cat(gts).numpy()
        y_pred = torch.cat(preds).numpy()
        meter["ba"] = balanced_acc(y_true, y_pred)
    return meter
