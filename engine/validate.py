from .metrics import balanced_acc
import torch, torch.nn as nn, torch.nn.functional as F
@torch.no_grad()
def validate(model, dl, device, class_weights_tri=None):
    model.eval()
    ce = nn.CrossEntropyLoss(weight=class_weights_tri).to(device)
    loss, preds, gts = 0.0, [], []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        y_tri = y[:, 0]
        tri_fused, *_ = model(x)
        loss += float(ce(tri_fused, y_tri).item()) * x.size(0)
        preds.append(tri_fused.argmax(-1).cpu()); gts.append(y_tri.cpu())
    loss /= len(dl.dataset)
    ba = balanced_acc(torch.cat(gts), torch.cat(preds))
    print("val_loss:", loss, "val_ba:", ba)
    return {"val_loss": loss, "val_ba": ba}
