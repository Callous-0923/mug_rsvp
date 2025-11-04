import torch, torch.nn.functional as F

def enable_mc_dropout(model):
    # 让 Dropout 在 eval 模式也生效
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

@torch.no_grad()
def mc_dropout_uncertainty(model, x, passes=10):
    """
    返回 EEG/EM 两个分支的不确定性u_eeg/u_eye（基于预测分布熵或方差）
    """
    if passes <= 0:
        zeros = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return zeros, zeros
    was_training = model.training
    model.eval()
    enable_mc_dropout(model)
    prob_eeg, prob_eye = [], []
    for _ in range(passes):
        tri_fused, tri_eeg, tri_eye, _, _ = model(x)
        prob_eeg.append(F.softmax(tri_eeg, dim=-1))
        prob_eye.append(F.softmax(tri_eye, dim=-1))
    P_eeg = torch.stack(prob_eeg, dim=0).mean(0)
    P_eye = torch.stack(prob_eye, dim=0).mean(0)
    # 熵作为不确定性（也可用方差）
    H_eeg = -(P_eeg*torch.log(P_eeg+1e-8)).sum(-1)
    H_eye = -(P_eye*torch.log(P_eye+1e-8)).sum(-1)
    # 归一化到0-1
    u_eeg = (H_eeg / torch.log(torch.tensor(P_eeg.shape[-1], device=H_eeg.device)))
    u_eye = (H_eye / torch.log(torch.tensor(P_eye.shape[-1], device=H_eye.device)))
    if was_training:
        model.train()
    else:
        model.eval()
    return u_eeg.clamp(0,1), u_eye.clamp(0,1)
