from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Optional


def temporal_smooth_loss(seq, weight=1.0):
    """
        时间平滑损失：对时间差分做 MSE（或 L2），支持 batch/time/... 形状
        seq: (B, T, C)
        返回标量
    """
    if seq.size(1) <= 1:
        return torch.tensor(0.0, device=seq.device, dtype=seq.dtype)
    diff = seq[:, 1:] - seq[:, :-1]
    return weight * (diff ** 2).mean()

def temporal_smooth_loss_2(seq, weight=1.0, w_s1=1.0, w_s2=0.2):
    """
        时间平滑损失：对时间差分做 MSE（或 L2）和二阶查分，支持 batch/time/... 形状
        seq: (B, T, C)
        返回标量
    """
    if seq.size(1) <= 1:
        return torch.tensor(0.0, device=seq.device, dtype=seq.dtype)
    diff = seq[:, 1:] - seq[:, :-1]
    loss_1st = (diff ** 2).mean()

    if seq.size(1) <= 2:
        return torch.tensor(0.0, device=seq.device, dtype=seq.dtype)

    diff2 = diff[:, 1:] - diff[:, :-1]
    loss_2nd = (diff2 ** 2).mean()

    # print("temporal_smooth_loss_2 - loss_1st:", loss_1st.item(), "loss_2nd:", loss_2nd.item())

    return weight * (w_s1 * loss_1st + w_s2 * loss_2nd)


class FMCycLoss(nn.Module):
    """
    base + condition-aware smooth V2
    激动 / 强情绪 / 快节奏时 → 允许不平滑
    平静时 → 强平滑
    """
    def __init__(self,
                 w_fm: float = 1.0,
                 w_emo: float = 1.0,
                 w_exp: float = 1.0,
                 w_rot: float = 10.0,
                 w_tran: float = 10.0,
                 w_smo: float = 10.0,
                 w_cyc: float = 1.0,
                 **kwargs):
        super(FMCycLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.w_fm = w_fm
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran
        self.w_smo = w_smo
        self.w_cyc = w_cyc

    def forward(self, gt_emotions_list, gt_3dmms_list, fm_outputs_list, **kwargs):
        """
        Args:
            gt_emotions_list: list of length B, each tensor (T, 25)
            gt_3dmms_list: list of length B, each tensor (T, 58)
            fm_outputs_list: list of dicts from decoder forward
                             each dict contains:
                                'v_pred': (B, T_window, z_dim)
                                'v_true': (B, T_window, z_dim)
                                'recon': (B, T_window, 58)
                                'listener_emotion': (B, T_window, 25)
        """
        total_fm_loss = 0.0
        total_rec_loss = 0.0
        total_rec_emotion_loss = 0.0
        total_rec_param_loss = 0.0
        window_start = 0

        if not fm_outputs_list:
            raise ValueError("No flow windows to supervise; check clip_length and window_size")

        for fm_outputs in fm_outputs_list:
            B, T_window, _ = fm_outputs['recon'].shape
            start = int(fm_outputs.get('window_start', window_start))
            end = start + T_window
            window_start = end

            # 对齐每个 batch 的 ground truth
            gt_3dmm_window = []
            gt_emotion_window = []
            for b in range(B):
                if len(gt_3dmms_list[b]) < end or len(gt_emotions_list[b]) < end:
                    raise ValueError(f"Ground truth for sample {b} is shorter than flow window [{start}:{end}]")
                gt_3dmm_window.append(gt_3dmms_list[b][start:end].to(fm_outputs['recon'].device))
                gt_emotion_window.append(gt_emotions_list[b][start:end].to(fm_outputs['listener_emotion'].device))

            gt_3dmm_window = torch.stack(gt_3dmm_window, dim=0)     # (B, T_window, 58)
            # gt_emotion_window = torch.stack(gt_emotion_window, dim=0)  # (B, T_window, 25)

            v_pred = fm_outputs["v_pred"]
            v_true = fm_outputs["v_true"]
            pred_3dmm = fm_outputs["recon"]
            # pred_emotion = fm_outputs["listener_emotion"]

            loss_cyc = self.w_cyc * fm_outputs["loss_cyc"]

            # Flow Matching loss
            fm_loss = self.w_fm * self.mse(v_pred, v_true)

            # print("fm_loss:", fm_loss.item())
            # print("loss_cyc:", loss_cyc.item())

            # 3DMM reconstruction loss
            exp_part = self.w_exp * self.mse(pred_3dmm[:, :, :52], gt_3dmm_window[:, :, :52])
            rot_part = self.w_rot * self.mse(pred_3dmm[:, :, 52:55], gt_3dmm_window[:, :, 52:55])
            tran_part = self.w_tran * self.mse(pred_3dmm[:, :, 55:], gt_3dmm_window[:, :, 55:])
            smooth_rot = self.w_smo * temporal_smooth_loss_2(pred_3dmm[:, :, 52:55], weight=10.0, w_s1=1.0, w_s2=0.5)
            smooth_tran = self.w_smo * temporal_smooth_loss_2(pred_3dmm[:, :, 55:], weight=5.0, w_s1=1.0, w_s2=0.5)

            # print("exp_part:", exp_part.item(), "rot_part:", rot_part.item(), "tran_part:", tran_part.item())
            # print("smooth_rot:", smooth_rot.item(), "smooth_tran:", smooth_tran.item())

            rec_param_loss = exp_part + rot_part + tran_part + smooth_rot + smooth_tran

            # # Emotion reconstruction
            # rec_emotion_loss = self.w_emo * self.mse(pred_emotion, gt_emotion_window)

            rec_emotion_loss = torch.tensor(0.0, device=pred_3dmm.device)

            rec_loss = rec_emotion_loss + rec_param_loss  + loss_cyc

            total_fm_loss += fm_loss
            total_rec_loss += rec_loss
            total_rec_emotion_loss += rec_emotion_loss
            total_rec_param_loss += rec_param_loss

        n = max(1, len(fm_outputs_list))
        total_fm_loss /= n
        total_rec_loss /= n
        total_rec_emotion_loss /= n
        total_rec_param_loss /= n

        total_loss = total_fm_loss + total_rec_loss

        return {
            "loss": total_loss,
            "loss_rec": total_rec_loss,
            "loss_emo": total_rec_emotion_loss,
            "loss_param": total_rec_param_loss,
            "loss_fm": total_fm_loss
        }

    def __repr__(self):
        return "FMCyc17Loss()：FMCyc15Loss smoothV2"
