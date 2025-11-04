# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .eeg_inception import EEGInceptionBackbone
from .eye_branch import EyeConvBackbone
from .fusion_mug import GatingFusion, contribution_weights, mug_weights
from .heads import TriHead, BinHead

class MUGRSVPNet(nn.Module):
    """
    最终多模态模型：
      EEGInceptionBackbone(384维) + EyeConvBackbone(128维)
      -> TriHead(EEG), TriHead(EYE), GatingFusion -> TriFused, BinHead
    """
    def __init__(self, eeg_chans=64, eye_chans=6, dropout=0.5):
        super().__init__()
        self.eeg_chans = eeg_chans
        self.eye_chans = eye_chans
        self.eeg = EEGInceptionBackbone(eeg_chans=eeg_chans, dropout=dropout)
        self.eye = EyeConvBackbone(eye_chans=eye_chans, dropout=dropout)
        self.tri_eeg = TriHead(384, 3)
        self.tri_eye = TriHead(128, 3)
        self.gate = GatingFusion(384+128)
        self.bin_head = BinHead(384+128, 2)

    def split_modal(self, x):
        # x: (B,1,C,T)
        eeg = x[:, :, :self.eeg_chans, :]
        eye = x[:, :, self.eeg_chans:self.eeg_chans+self.eye_chans, :]
        return eeg, eye

    def forward(self, x, w_mug=None):
        eeg_x, eye_x = self.split_modal(x)
        f_eeg, _mid = self.eeg(eeg_x)                    # (B,384)
        f_eye = self.eye(eye_x)                          # (B,128)
        tri_eeg = self.tri_eeg(f_eeg)                    # (B,3)
        tri_eye = self.tri_eye(f_eye)                    # (B,3)
        gates, fused_feat = self.gate(f_eeg, f_eye)      # gates:(B,2), fused_feat:(B,512)
        tri_fused = (torch.stack([tri_eeg, tri_eye], dim=1) * gates.unsqueeze(-1)).sum(1)
        bin_logits = self.bin_head(fused_feat)           # (B,2)

        # 返回所有用于损失/分析的logits与门控
        return tri_fused, tri_eeg, tri_eye, bin_logits, gates
