from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Dict, Optional


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class VAELoss(nn.Module):
    def __init__(self, kl_p: float = 0.0002,
                 w_emo: float = None,
                 w_exp: float = None,
                 w_rot: float = None,
                 w_tran: float = None,
                 **kwargs):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

        if w_emo is None:
            w_emo = 1.0
        if w_exp is None:
            w_exp = 1.0
        if w_rot is None:
            w_rot = 10.0
        if w_tran is None:
            w_tran = 10.0
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran

    def forward(self, gt_emotions, gt_3dmms, pred_emotions, pred_3dmms, distribution):
        """ List
        gt_emotion; gt_3dmm; pred_emotion; pred_3dmm
        """
        bsz = len(gt_emotions)
        rec_emotion_loss = 0
        rec_param_loss = 0
        for gt_emotion, gt_3dmm, pred_emotion, pred_3dmm in zip(
                gt_emotions, gt_3dmms, pred_emotions, pred_3dmms):
            gt_emotion = gt_emotion.to(pred_emotion.get_device())
            gt_3dmm = gt_3dmm.to(pred_3dmm.get_device())

            exp_part = self.w_exp * self.mse(pred_3dmm[:, :52], gt_3dmm[:, :52])  # expression
            rot_part = self.w_rot * self.mse(pred_3dmm[:, 52:55], gt_3dmm[:, 52:55])  # rotation
            tran_part = self.w_tran * self.mse(pred_3dmm[:, 55:], gt_3dmm[:, 55:])  # translation
            rec_param_loss = rec_param_loss + (exp_part + rot_part + tran_part)
            rec_emotion_loss = rec_emotion_loss + self.w_emo * self.mse(pred_emotion, gt_emotion)

        rec_emotion_loss = rec_emotion_loss / bsz
        rec_param_loss = rec_param_loss / bsz
        rec_loss = rec_emotion_loss + rec_param_loss

        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss
        return loss, rec_loss, rec_emotion_loss, rec_param_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"


def div_loss(Y_1_list, Y_2_list):
    loss = 0.0
    B = len(Y_1_list)
    for y1, y2 in zip(Y_1_list, Y_2_list):
        y1_flat = y1.view(1, -1)
        y2_flat = y2.view(1, -1)
        Y = torch.cat([y1_flat, y2_flat], dim=0)
        dist2 = F.pdist(Y, 2) ** 2
        loss += (-dist2 / 100).exp().mean()
    loss = loss / B
    return loss


def div_loss_v2(Y_1, Y_2):
    loss = 0.0
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss


def TemporalLoss(Y):
    diff = Y[:, 1:, :] - Y[:, :-1, :]
    t_loss = torch.mean(torch.norm(diff, dim=2, p=2) ** 2)
    return t_loss


def L1Loss(prediction, target, reduction="min", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(dim=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(dim=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELoss(prediction, target, reduction="mean", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    loss = ((prediction - target) ** 2).mean(dim=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(dim=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELossApt(prediction, target, reduction="mean",
               w_au=1.0, w_va=5.0, w_em=2.0, **kwargs):
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"
    loss_au = F.mse_loss(prediction[:, :, :15], target[:, :, :15], reduction=reduction)
    loss_va = F.mse_loss(prediction[:, :, 15:17], target[:, :, 15:17], reduction=reduction)
    loss_em = F.mse_loss(prediction[:, :, 17:], target[:, :, 17:], reduction=reduction)
    loss = loss_au * w_au + loss_va * w_va + loss_em * w_em

    losses_dict = {"loss": loss, "loss_au": loss_au, "loss_va": loss_va, "loss_em": loss_em}
    return losses_dict


class DiffusionLoss:
    def __init__(self,
                 losses_type='MSELoss',
                 n_preds=10,
                 w_au=1.0,  # action unit
                 w_va=5.0,  # valence and arousal
                 w_em=2.0,  # emotion
                 **kwargs):
        self.loss_type = losses_type
        self.n_preds = n_preds
        self.w_au = w_au
        self.w_va = w_va
        self.w_em = w_em

    def __call__(self, output_decoder):
        prediction_emotion = output_decoder["prediction_emotion"]
        target_emotion = output_decoder["target_emotion"]

        _, _, window_size, emotion_dim = prediction_emotion.shape
        prediction_emotion = prediction_emotion.reshape(-1, self.n_preds, window_size * emotion_dim)
        target_emotion = target_emotion.reshape(-1, self.n_preds, window_size * emotion_dim)
        losses_dict = eval(self.loss_type)(
            prediction_emotion, target_emotion, k=self.n_preds, w_au=self.w_au, w_va=self.w_va, w_em=self.w_em)

        return losses_dict


class FlowMatchingDiffusionLoss:
    def __init__(self,
                 w_fm: float = 1.0,
                 w_exp: float = 2.0,
                 w_rot: float = 4.0,
                 w_tran: float = 4.0,
                 w_smo: float = 10.0,
                 w_au: float = 1.0,
                 w_va: float = 5.0,
                 w_em: float = 5.0,
                 **kwargs):
        self.w_fm = w_fm
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran
        self.w_smo = w_smo
        self.w_au = w_au
        self.w_va = w_va
        self.w_em = w_em

    @staticmethod
    def _expand_target(target, prediction):
        if target.shape[1] == prediction.shape[1]:
            return target
        if target.shape[1] == 1:
            return target.expand(-1, prediction.shape[1], -1, -1)
        raise ValueError(f"Cannot align target shape {target.shape} to prediction shape {prediction.shape}")

    @staticmethod
    def _masked_mse(prediction, target, mask=None):
        if mask is None:
            return F.mse_loss(prediction, target)
        while mask.dim() < prediction.dim():
            mask = mask.unsqueeze(1)
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)
        mask = mask.expand(*prediction.shape[:-1], 1)
        sq = (prediction - target) ** 2
        denom = mask.sum() * prediction.shape[-1]
        return (sq * mask).sum() / denom.clamp_min(1.0)

    @staticmethod
    def _smooth_loss(seq):
        if seq.size(-2) <= 1:
            return torch.tensor(0.0, device=seq.device, dtype=seq.dtype)
        diff = seq[..., 1:, :] - seq[..., :-1, :]
        return (diff ** 2).mean()

    def __call__(self, output_decoder):
        prediction_emotion = output_decoder["prediction_emotion"]
        target_emotion = self._expand_target(output_decoder["target_emotion"], prediction_emotion)
        prediction_3dmm = output_decoder["prediction_3dmm"]
        target_3dmm = self._expand_target(output_decoder["target_3dmm"], prediction_3dmm)
        motion_mask = output_decoder.get("motion_mask")

        loss_au = self._masked_mse(prediction_emotion[..., :15], target_emotion[..., :15], motion_mask)
        loss_va = self._masked_mse(prediction_emotion[..., 15:17], target_emotion[..., 15:17], motion_mask)
        loss_em = self._masked_mse(prediction_emotion[..., 17:], target_emotion[..., 17:], motion_mask)
        loss_emo = self.w_au * loss_au + self.w_va * loss_va + self.w_em * loss_em

        loss_exp = self.w_exp * self._masked_mse(prediction_3dmm[..., :52], target_3dmm[..., :52], motion_mask)
        loss_rot = self.w_rot * self._masked_mse(prediction_3dmm[..., 52:55], target_3dmm[..., 52:55], motion_mask)
        loss_tran = self.w_tran * self._masked_mse(prediction_3dmm[..., 55:], target_3dmm[..., 55:], motion_mask)
        loss_smo = self.w_smo * (
            self._smooth_loss(prediction_3dmm[..., 52:55]) +
            self._smooth_loss(prediction_3dmm[..., 55:])
        )
        loss_param = loss_exp + loss_rot + loss_tran + loss_smo

        fm_outputs = output_decoder.get("fm_outputs", [])
        if len(fm_outputs) == 0:
            loss_fm = torch.tensor(0.0, device=prediction_3dmm.device, dtype=prediction_3dmm.dtype)
        else:
            fm_losses = []
            for fm_output in fm_outputs:
                fm_losses.append(
                    self._masked_mse(
                        fm_output["v_pred"],
                        fm_output["v_true"],
                        fm_output.get("mask"),
                    )
                )
            loss_fm = torch.stack(fm_losses).mean()

        loss_rec = loss_emo + loss_param
        loss = self.w_fm * loss_fm + loss_rec
        return {
            "loss": loss,
            "loss_fm": loss_fm,
            "loss_param": loss_param,
            "loss_au": loss_au,
            "loss_va": loss_va,
            "loss_em": loss_em,
            "loss_emo": loss_emo,
            "loss_rec": loss_rec,
        }

class EmotionVAELoss:
    def __init__(self, w_au, w_va, w_em, w_kld, **kwargs):
        self.w_au = w_au
        self.w_va = w_va
        self.w_em = w_em
        self.w_kld = w_kld

        self.au_criterion = nn.BCEWithLogitsLoss(reduction="none")  # "mean"
        self.va_criterion = nn.MSELoss(reduction="mean")
        self.em_criterion = nn.KLDivLoss(reduction="none")  # batchmean
        self.kld_criterion = KLLoss()

    def kld_loss(self, mu, logvar, reduction='batchmean'):
        kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
        if reduction == 'sum':
            # sum over B, L, D
            return -0.5 * torch.sum(kld_element)
        elif reduction == 'batchmean':
            # sum over L,D, then mean over batch
            kld = -0.5 * torch.sum(kld_element, dim=(1, 2))  # [B]
            return torch.mean(kld)  # scalar
        elif reduction == 'mean':
            # mean over all elements
            return -0.5 * torch.mean(kld_element)
        else:
            raise ValueError("Unknown reduction")

    def __call__(self, predictions, targets, distribution, mask=None):
        au_logits, va_logits, emotion_logits = predictions
        au_targets, va_targets, emotion_targets = \
            targets[:, :, :15], targets[:, :, 15:17], targets[:, :, 17:]

        au_loss = self.au_criterion(au_logits, au_targets)  # AUs
        va_loss = self.va_criterion(va_logits, va_targets)  # valence and arousal
        emotion_logits = rearrange(F.log_softmax(emotion_logits, dim=-1), "b l d -> (b l) d")
        emotion_targets = rearrange(emotion_targets, "b l d -> (b l) d")
        em_loss = self.em_criterion(emotion_logits, emotion_targets)

        if mask is not None:
            au_mask = mask  # [B, L, 1]
            au_loss = (au_loss * au_mask).sum() / (au_mask.sum() * au_logits.shape[-1])
            em_mask = rearrange(mask, "b l d -> (b l) d")  # [BxL, 1]
            em_loss = (em_loss * em_mask).sum() / em_mask.sum()

        mu_ref = torch.zeros_like(distribution.loc).to(au_logits)
        scale_ref = torch.ones_like(distribution.scale).to(au_logits)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kld_loss = self.kld_criterion(distribution, distribution_ref)

        loss = self.w_kld * kld_loss + \
               self.w_au * au_loss + \
               self.w_va * va_loss + \
               self.w_em * em_loss

        return loss, kld_loss, au_loss, va_loss, em_loss



class FMLoss(nn.Module):
    def __init__(self,
                 w_fm: float = 1.0,
                 w_emo: float = 1.0,
                 w_exp: float = 1.0,
                 w_rot: float = 10.0,
                 w_tran: float = 10.0,
                 w_smo: float = 10.0,
                 **kwargs):
        super(FMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.w_fm = w_fm
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran
        self.w_smo = w_smo

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

        for fm_outputs in fm_outputs_list:
            B, T_window, _ = fm_outputs['recon'].shape

            # 对齐每个 batch 的 ground truth
            gt_3dmm_window = []
            gt_emotion_window = []
            for b in range(B):
                gt_3dmm_window.append(gt_3dmms_list[b][:T_window].to(fm_outputs['recon'].device))
                gt_emotion_window.append(gt_emotions_list[b][:T_window].to(fm_outputs['listener_emotion'].device))

            gt_3dmm_window = torch.stack(gt_3dmm_window, dim=0)     # (B, T_window, 58)
            gt_emotion_window = torch.stack(gt_emotion_window, dim=0)  # (B, T_window, 25)

            v_pred = fm_outputs["v_pred"]
            v_true = fm_outputs["v_true"]
            pred_3dmm = fm_outputs["recon"]
            pred_emotion = fm_outputs["listener_emotion"]

            # Flow Matching loss
            fm_loss = self.mse(v_pred, v_true)

            # 3DMM reconstruction loss
            exp_part = self.w_exp * self.mse(pred_3dmm[:, :, :52], gt_3dmm_window[:, :, :52])
            rot_part = self.w_rot * self.mse(pred_3dmm[:, :, 52:55], gt_3dmm_window[:, :, 52:55])
            tran_part = self.w_tran * self.mse(pred_3dmm[:, :, 55:], gt_3dmm_window[:, :, 55:])
            smooth_rot = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 52:55], weight=10.0)
            smooth_tran = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 55:], weight=5.0)

            # print("exp_part:", exp_part.item(), "rot_part:", rot_part.item(), "tran_part:", tran_part.item())
            # print("smooth_rot:", smooth_rot.item(), "smooth_tran:", smooth_tran.item())

            rec_param_loss = exp_part + rot_part + tran_part + smooth_rot + smooth_tran

            # Emotion reconstruction
            rec_emotion_loss = self.w_emo * self.mse(pred_emotion, gt_emotion_window)

            rec_loss = rec_emotion_loss + rec_param_loss

            total_fm_loss += fm_loss
            total_rec_loss += rec_loss
            total_rec_emotion_loss += rec_emotion_loss
            total_rec_param_loss += rec_param_loss

        n = max(1, len(fm_outputs_list))
        total_fm_loss /= n
        total_rec_loss /= n
        total_rec_emotion_loss /= n
        total_rec_param_loss /= n

        total_loss = self.w_fm * total_fm_loss + total_rec_loss

        return {
            "loss": total_loss,
            "loss_rec": total_rec_loss,
            "loss_emo": total_rec_emotion_loss,
            "loss_param": total_rec_param_loss,
            "loss_fm": total_fm_loss
        }

    def __repr__(self):
        return "FMLoss()"


class FMAddEyeLoss(nn.Module):
    def __init__(self,
                 w_fm: float = 1.0,
                 w_emo: float = 1.0,
                 w_exp: float = 1.0,
                 w_rot: float = 10.0,
                 w_tran: float = 10.0,
                 w_smo: float = 10.0,
                 w_add_eye: float = 2.0,
                 **kwargs):
        super(FMAddEyeLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.w_fm = w_fm
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran
        self.w_smo = w_smo
        self.w_add_eye = w_add_eye

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

        for fm_outputs in fm_outputs_list:
            B, T_window, _ = fm_outputs['recon'].shape

            # 对齐每个 batch 的 ground truth
            gt_3dmm_window = []
            gt_emotion_window = []
            for b in range(B):
                gt_3dmm_window.append(gt_3dmms_list[b][:T_window].to(fm_outputs['recon'].device))
                gt_emotion_window.append(gt_emotions_list[b][:T_window].to(fm_outputs['listener_emotion'].device))

            gt_3dmm_window = torch.stack(gt_3dmm_window, dim=0)     # (B, T_window, 58)
            gt_emotion_window = torch.stack(gt_emotion_window, dim=0)  # (B, T_window, 25)

            v_pred = fm_outputs["v_pred"]
            v_true = fm_outputs["v_true"]
            pred_3dmm = fm_outputs["recon"]
            pred_emotion = fm_outputs["listener_emotion"]

            # Flow Matching loss
            fm_loss = self.mse(v_pred, v_true)

            # 3DMM reconstruction loss
            exp_part = self.w_exp * self.mse(pred_3dmm[:, :, :52], gt_3dmm_window[:, :, :52])
            rot_part = self.w_rot * self.mse(pred_3dmm[:, :, 52:55], gt_3dmm_window[:, :, 52:55])
            tran_part = self.w_tran * self.mse(pred_3dmm[:, :, 55:], gt_3dmm_window[:, :, 55:])
            smooth_rot = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 52:55], weight=10.0)
            smooth_tran = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 55:], weight=5.0)

            add_eye_part = self.w_add_eye * self.mse(pred_3dmm[:, :, :14], gt_3dmm_window[:, :, :14])

            # print("exp_part:", exp_part.item(), "rot_part:", rot_part.item(), "tran_part:", tran_part.item())
            # print("smooth_rot:", smooth_rot.item(), "smooth_tran:", smooth_tran.item())

            rec_param_loss = exp_part + rot_part + tran_part + smooth_rot + smooth_tran + add_eye_part

            # Emotion reconstruction
            rec_emotion_loss = self.w_emo * self.mse(pred_emotion, gt_emotion_window)

            rec_loss = rec_emotion_loss + rec_param_loss

            total_fm_loss += fm_loss
            total_rec_loss += rec_loss
            total_rec_emotion_loss += rec_emotion_loss
            total_rec_param_loss += rec_param_loss

        n = max(1, len(fm_outputs_list))
        total_fm_loss /= n
        total_rec_loss /= n
        total_rec_emotion_loss /= n
        total_rec_param_loss /= n

        total_loss = self.w_fm * total_fm_loss + total_rec_loss

        return {
            "loss": total_loss,
            "loss_rec": total_rec_loss,
            "loss_emo": total_rec_emotion_loss,
            "loss_param": total_rec_param_loss,
            "loss_fm": total_fm_loss
        }

    def __repr__(self):
        return "FMAddEyeLoss()"



class EyeFMLoss(nn.Module):
    def __init__(self,
                 w_fm: float = 1.0,
                 w_emo: float = 1.0,
                 w_exp: float = 1.0,
                 w_rot: float = 10.0,
                 w_tran: float = 10.0,
                 w_smo: float = 10.0,
                 w_phase: float = 0.2,
                 **kwargs):
        super(EyeFMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.w_fm = w_fm
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran
        self.w_smo = w_smo
        self.w_phase = w_phase

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

        for fm_outputs in fm_outputs_list:
            B, T_window, _ = fm_outputs['recon'].shape

            # 对齐每个 batch 的 ground truth
            gt_3dmm_window = []
            gt_emotion_window = []
            for b in range(B):
                gt_3dmm_window.append(gt_3dmms_list[b][:T_window].to(fm_outputs['recon'].device))
                gt_emotion_window.append(gt_emotions_list[b][:T_window].to(fm_outputs['listener_emotion'].device))

            gt_3dmm_window = torch.stack(gt_3dmm_window, dim=0)     # (B, T_window, 58)
            gt_emotion_window = torch.stack(gt_emotion_window, dim=0)  # (B, T_window, 25)

            v_pred = fm_outputs["v_pred"]
            v_true = fm_outputs["v_true"]
            pred_3dmm = fm_outputs["recon"]
            pred_emotion = fm_outputs["listener_emotion"]
            sin_phi = fm_outputs["sin_phi"]
            cos_phi = fm_outputs["cos_phi"]

            L_phase_norm = ((sin_phi ** 2 + cos_phi ** 2 - 1.0) ** 2).mean()

            # Flow Matching loss
            fm_loss = self.mse(v_pred, v_true) + self.w_phase * L_phase_norm

            # print("FM loss:", fm_loss.item())
            # print("L_phase_norm:",  self.w_phase * L_phase_norm.item())

            # 3DMM reconstruction loss
            exp_part = self.w_exp * self.mse(pred_3dmm[:, :, :52], gt_3dmm_window[:, :, :52])
            rot_part = self.w_rot * self.mse(pred_3dmm[:, :, 52:55], gt_3dmm_window[:, :, 52:55])
            tran_part = self.w_tran * self.mse(pred_3dmm[:, :, 55:], gt_3dmm_window[:, :, 55:])
            smooth_rot = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 52:55], weight=10.0)
            smooth_tran = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 55:], weight=5.0)

            # print("exp_part:", exp_part.item(), "rot_part:", rot_part.item(), "tran_part:", tran_part.item())
            # print("smooth_rot:", smooth_rot.item(), "smooth_tran:", smooth_tran.item())

            rec_param_loss = exp_part + rot_part + tran_part + smooth_rot + smooth_tran

            # Emotion reconstruction
            rec_emotion_loss = self.w_emo * self.mse(pred_emotion, gt_emotion_window)

            rec_loss = rec_emotion_loss + rec_param_loss

            total_fm_loss += fm_loss
            total_rec_loss += rec_loss
            total_rec_emotion_loss += rec_emotion_loss
            total_rec_param_loss += rec_param_loss

        n = max(1, len(fm_outputs_list))
        total_fm_loss /= n
        total_rec_loss /= n
        total_rec_emotion_loss /= n
        total_rec_param_loss /= n

        total_loss = self.w_fm * total_fm_loss + total_rec_loss

        return {
            "loss": total_loss,
            "loss_rec": total_rec_loss,
            "loss_emo": total_rec_emotion_loss,
            "loss_param": total_rec_param_loss,
            "loss_fm": total_fm_loss
        }

    def __repr__(self):
        return "EyeFMLoss()"



def div_loss_flow(Y_1, Y_2, scale: float = 100.0):
    """
    Diversity loss between two model outputs.
    Args:
        Y_1: Tensor [B, T, C]
        Y_2: Tensor [B, T, C]
        scale: denominator for distance normalization
    """
    # print("Y_1 shape:", Y_1.shape)
    # print("Y_2 shape:", Y_2.shape)
    assert Y_1.shape == Y_2.shape, "Shapes of Y_1 and Y_2 must match"

    B = Y_1.size(0)
    loss = 0.0
    for b in range(B):
        y1 = Y_1[b].reshape(1, -1)  # [1, T*C]
        y2 = Y_2[b].reshape(1, -1)  # [1, T*C]
        Y = torch.cat([y1, y2], dim=0)  # [2, T*C]
        dist2 = F.pdist(Y, 2) ** 2
        loss += torch.exp(-dist2 / scale).mean()
    return loss / B

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


class EmotionLoss(nn.Module):
    """
    CLIP-style 双塔跨模态对比损失：
        - 输入：3dmm_emb, label_emb 形状均为 (N, embed_dim)
        - 输出：标量 loss
    要求输入已经 L2-normalized（你的模型中已处理）
    """
    def __init__(self, temperature=0.07, learnable_temperature=True, scale=1, **kwargs):
        super().__init__()
        self.scale = scale
        if learnable_temperature:
            # 使用 log_temp 避免数值不稳定
            self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, emb_3dmm, emb_label, **kwargs):
        """
        emb_3dmm: (N, D)
        emb_label: (N, D)
        """
        # 计算相似度矩阵 (N, N)
        # 两侧已 L2-normalized → 直接点积 = 余弦相似度
        logits = emb_3dmm @ emb_label.t()  # (N, N)

        # 温度缩放
        temp = torch.exp(self.log_temp)
        logits = logits * temp

        # 正样本在对角线
        targets = torch.arange(logits.size(0), device=logits.device)

        # 3DMM → Label 对齐损失
        loss_3dmm2label = F.cross_entropy(logits, targets)

        # Label → 3DMM 对齐损失（对称性）
        loss_label23dmm = F.cross_entropy(logits.t(), targets)

        loss = (loss_3dmm2label + loss_label23dmm) * 0.5 * self.scale
        return {
            "loss": loss,
        }

    def __repr__(self):
        return "EmotionLoss()"


class EnhancedFMLoss(nn.Module):
    """
    增强的 Flow-Matching 训练损失：
      - 基础 Flow Matching loss (v_pred vs v_true)
      - 重建损失：3DMM（exp/rot/tran） + 时间平滑
      - 情感重建（MSE）
      - 可选：多样性鼓励 loss (鼓励生成样本之间差异)
      - 可选：embedding 对比损失 (InfoNCE-style)，需要 fm_outputs 中包含：
            - 'gen_reaction_emb' : (B, K, D_emb)  或 (B*K, D_emb) (我们支持两者)
            - 'z_label' : (B, D_emb)  （正例 embedding）
            - 'possible_labels_emb' : (B, Nposs, D_emb)  （负例候选）
      - 对缺失键/信息做容错跳过

    返回 dict 包含分项与总 loss，方便日志记录与权重调优。
    """

    def __init__(self,
                 w_fm: float = 1.0,
                 w_emo: float = 1.0,
                 w_exp: float = 1.0,
                 w_rot: float = 10.0,
                 w_tran: float = 10.0,
                 w_smo: float = 10.0,
                 w_diverse: float = 1.0,
                 div_margin: float = 0.8,
                 w_contrastive: float = 1.0,
                 contrastive_temp: float = 0.07,
                 eps: float = 1e-8,
                 reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.w_fm = w_fm
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran
        self.w_smo = w_smo
        self.w_diverse = w_diverse
        self.div_margin = div_margin
        self.w_contrastive = w_contrastive
        self.contrastive_temp = contrastive_temp
        self.eps = float(eps)

    def _pairwise_mean_l2(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算一组向量的平均 pairwise L2 距离
        x: (..., K, D) or (K, D)
        返回标量 tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,K,D)
        B, K, D = x.shape
        if K <= 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # (B, K, K, D)
        x_i = x.unsqueeze(2)  # (B,K,1,D)
        x_j = x.unsqueeze(1)  # (B,1,K,D)
        diffs = x_i - x_j
        dists = torch.sqrt(torch.clamp((diffs * diffs).sum(-1), min=self.eps))  # (B,K,K)
        # take upper triangle without diagonal
        mask = torch.triu(torch.ones(K, K, device=x.device), diagonal=1).bool()
        dists_ut = dists[:, mask].view(B, -1)
        mean_dist = dists_ut.mean()
        return mean_dist

    def _info_nce_loss(self,
                       anchor: torch.Tensor,
                       positive: torch.Tensor,
                       negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        简化的 InfoNCE:
          - anchor: (B, D)
          - positive: (B, D)
          - negatives: (B, Nneg, D)  (optional)
        返回平均负对数似然标量（越小越好）。
        """
        # normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        sim_pos = torch.sum(anchor * positive, dim=-1) / self.contrastive_temp  # (B,)

        if negatives is None:
            # use other anchors in batch as negatives
            negatives = anchor.unsqueeze(1)  # (B,1,D) -> will be masked below
            # build logits with pos and other anchors
            # simpler fallback: maximize anchor-pos cosine (use -sim for loss)
            loss = -torch.mean(sim_pos)
            return loss

        # negatives: (B, Nneg, D)
        negatives = F.normalize(negatives, dim=-1)
        sim_negs = torch.einsum('bd,bnd->bn', anchor, negatives) / self.contrastive_temp  # (B, Nneg)

        logits = torch.cat([sim_pos.unsqueeze(1), sim_negs], dim=1)  # (B, 1+Nneg)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positive index 0
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self,
                gt_emotions_list: List[torch.Tensor],
                gt_3dmms_list: List[torch.Tensor],
                fm_outputs_list: List[Dict], **kwargs):
        """
        参数同你现有的 FMLoss，但支持更多键。
        fm_outputs_list: list of dicts, 每个 dict 可能包含以下键（部分可选）:
          - 'v_pred', 'v_true'            : 流匹配项
          - 'recon'                      : 重建 3DMM (B,T,58)
          - 'listener_emotion'           : 预测情感 (B,T,25)
          - 'gen_reactions_diverse'      : (B, K, T, 58)  可选（训练时生成样本）
          - 'gen_emotions_diverse'       : (B, K, T, 25)  可选
          - 'gen_reaction_emb'           : (B, K, D_emb) or (B*K, D_emb)
          - 'z_label'                    : (B, D_emb)  正例 embedding（可选）
          - 'possible_labels_emb'        : (B, Nposs, D_emb)  候选负例（可选）
        """
        total_fm_loss = 0.0
        total_rec_loss = 0.0
        total_rec_emotion_loss = 0.0
        total_rec_param_loss = 0.0
        total_div_loss = 0.0
        total_contrastive_loss = 0.0

        device = None

        for fm_outputs in fm_outputs_list:
            # device infer
            if device is None:
                some = next(iter(fm_outputs.values()))
                device = some.device if hasattr(some, 'device') else None

            # ---- prepare GT windows (align) ----
            # handle missing 'recon' gracefully
            if 'recon' in fm_outputs:
                B, T_window, _ = fm_outputs['recon'].shape
            else:
                # if no recon present, try to infer from v_pred shape
                if 'v_pred' in fm_outputs:
                    B, T_window, _ = fm_outputs['v_pred'].shape
                else:
                    # skip this fm_outputs if cannot infer temporal size
                    continue

            # build gt windows per-batch element
            gt_3dmm_window = []
            gt_emotion_window = []
            for b in range(B):
                gt_3dmm_window.append(gt_3dmms_list[b][:T_window].to(device))
                gt_emotion_window.append(gt_emotions_list[b][:T_window].to(device))
            gt_3dmm_window = torch.stack(gt_3dmm_window, dim=0)  # (B,T_window,58)
            gt_emotion_window = torch.stack(gt_emotion_window, dim=0)  # (B,T_window,25)

            # ---- Flow Matching loss ----
            fm_loss = torch.tensor(0.0, device=device)
            if 'v_pred' in fm_outputs and 'v_true' in fm_outputs:
                fm_loss = self.mse(fm_outputs['v_pred'], fm_outputs['v_true'])
            total_fm_loss = total_fm_loss + fm_loss

            # ---- Reconstruction losses (3DMM param breakdown) ----
            rec_param_loss = torch.tensor(0.0, device=device)
            if 'recon' in fm_outputs:
                pred_3dmm = fm_outputs['recon']
                # expression (first 52)
                exp_part = self.w_exp * self.mse(pred_3dmm[:, :, :52], gt_3dmm_window[:, :, :52])
                # rotation (52:55) 可能为 axis-angle 或 euler
                rot_part = self.w_rot * self.mse(pred_3dmm[:, :, 52:55], gt_3dmm_window[:, :, 52:55])
                # translation (55:)
                tran_part = self.w_tran * self.mse(pred_3dmm[:, :, 55:], gt_3dmm_window[:, :, 55:])
                smooth_rot = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 52:55], weight=1.0)
                smooth_tran = self.w_smo * temporal_smooth_loss(pred_3dmm[:, :, 55:], weight=1.0)
                rec_param_loss = exp_part + rot_part + tran_part + smooth_rot + smooth_tran
                # print("exp_part:", exp_part.item(), "rot_part:", rot_part.item(), "tran_part:", tran_part.item())
                # print("smooth_rot:", smooth_rot.item(), "smooth_tran:", smooth_tran.item())

            # ---- Emotion reconstruction ----
            rec_emotion_loss = torch.tensor(0.0, device=device)
            if 'listener_emotion' in fm_outputs:
                rec_emotion_loss = self.w_emo * self.mse(fm_outputs['listener_emotion'], gt_emotion_window)

            rec_loss = rec_emotion_loss + rec_param_loss

            total_rec_loss = total_rec_loss + rec_loss
            total_rec_emotion_loss = total_rec_emotion_loss + rec_emotion_loss
            total_rec_param_loss = total_rec_param_loss + rec_param_loss

            # ---- Diversity loss (鼓励多样化样本互相不同) ----
            div_loss = torch.tensor(0.0, device=device)
            if 'gen_reactions_diverse' in fm_outputs:
                # gen_reactions_diverse: (B, K, T, 58)
                gen = fm_outputs['gen_reactions_diverse']
                B_gen, K, T_g, Dg = gen.shape
                # flatten temporal维到向量表示再计算 pairwise
                # 方式1: 在每个样本上做时间平均再 pairwise
                # gen_flat = gen.mean(dim=2)  # (B, K, Dg)
                gen_flat = F.normalize(gen.mean(dim=2), dim=-1)
                mean_pairwise = self._pairwise_mean_l2(gen_flat)
                div_loss = self.w_diverse * F.relu(self.div_margin - mean_pairwise)

                # mean_pairwise = self._pairwise_mean_l2(gen_flat)  # scalar
                # # 采用负均值距离作为 loss（因为优化器最小化 loss，我们要**最大化**距离）
                # div_loss = - self.w_diverse * mean_pairwise
            total_div_loss = total_div_loss + div_loss

            # ---- Contrastive / label alignment loss using embeddings ----
            contrast_loss = torch.tensor(0.0, device=device)
            # 需要 'gen_reaction_emb' 和 'z_label' 才能计算
            if 'gen_reaction_emb' in fm_outputs and 'z_label' in fm_outputs:
                gen_emb = fm_outputs['gen_reaction_emb']  # maybe (B,K,D) or (B*K,D)
                z_label = fm_outputs['z_label']  # (B,D)
                possible_emb = fm_outputs.get('possible_labels_emb', None)  # (B,Nposs,D) or None

                # normalize shapes: ensure gen_emb -> (B,K,D)
                if gen_emb.dim() == 2:
                    # assume (B*K, D) -> reshape if K can be inferred from fm_outputs['gen_reactions_diverse']
                    if 'gen_reactions_diverse' in fm_outputs:
                        K = fm_outputs['gen_reactions_diverse'].shape[1]
                        gen_emb = gen_emb.view(B, K, -1)
                    else:
                        # fallback: treat as single sample (K=1)
                        gen_emb = gen_emb.unsqueeze(1)

                # for each gen sample we compute anchor vs positive (z_label)
                # aggregate anchors by averaging K samples' embeddings into one anchor per batch
                anchor = gen_emb.mean(dim=1)  # (B, D)
                pos = z_label  # (B, D)
                negs = possible_emb if possible_emb is not None else None
                contrast_loss = self.w_contrastive * self._info_nce_loss(anchor, pos, negs)

            total_contrastive_loss = total_contrastive_loss + contrast_loss

        # ---- 平均化 ----
        n = max(1, len(fm_outputs_list))
        total_fm_loss = total_fm_loss / n
        total_rec_loss = total_rec_loss / n
        total_rec_emotion_loss = total_rec_emotion_loss / n
        total_rec_param_loss = total_rec_param_loss / n
        total_div_loss = total_div_loss / n
        total_contrastive_loss = total_contrastive_loss / n

        total_loss = self.w_fm * total_fm_loss + total_rec_loss + total_div_loss + total_contrastive_loss

        # # 打印所有分项损失
        # print("EnhancedFMLoss components -- total: {:.6f}, fm: {:.6f}, rec: {:.6f} (emo: {:.6f}, param: {:.6f}), div: {:.6f}, contrastive: {:.6f}".format(
        #     total_loss.item(),
        #     total_fm_loss.item(),
        #     total_rec_loss.item(),
        #     total_rec_emotion_loss.item(),
        #     total_rec_param_loss.item(),
        #     total_div_loss.item(),
        #     total_contrastive_loss.item()
        # ))

        return {
            "loss": total_loss,
            "loss_rec": total_rec_loss,
            "loss_emo": total_rec_emotion_loss,
            "loss_param": total_rec_param_loss,
            "loss_fm": total_fm_loss,
            "loss_diverse": total_div_loss,
            "loss_contrastive": total_contrastive_loss
        }

    def __repr__(self):
        return f"EnhancedFMLoss(w_fm={self.w_fm}, w_emo={self.w_emo}, w_exp={self.w_exp}, w_rot={self.w_rot}, w_tran={self.w_tran}, w_smo={self.w_smo}, w_diverse={self.w_diverse}, w_contrastive={self.w_contrastive})"


def temporal_smooth_loss_2(x, weight=1.0):
    """Second-order smoothness (acceleration penalty)"""
    if x.size(1) < 3:
        return torch.tensor(0.0, device=x.device)
    acc = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return weight * torch.mean(acc ** 2)


def fft_spectrum_loss(pred_expr, gt_expr, weight=0.1):
    """Encourage matching high-frequency spectrum (micro-motions)"""
    if pred_expr.size(1) < 8:
        return torch.tensor(0.0, device=pred_expr.device)
    pred_fft = torch.fft.rfft(pred_expr, dim=1)
    gt_fft = torch.fft.rfft(gt_expr, dim=1)
    pred_mag = pred_fft.abs()
    gt_mag = gt_fft.abs()
    n_bins = pred_mag.size(1)
    hf_start = int(n_bins * 0.7)
    if hf_start < n_bins:
        hf_loss = F.l1_loss(pred_mag[:, hf_start:], gt_mag[:, hf_start:])
    else:
        hf_loss = torch.tensor(0.0, device=pred_expr.device)
    return weight * hf_loss


class FMLossFFT(nn.Module):
    def __init__(self,
                 w_fm: float = 1.0,
                 w_emo: float = 1.0,
                 w_exp: float = 1.0,
                 w_rot: float = 10.0,
                 w_tran: float = 10.0,
                 w_smo: float = 5.0,      # ↓ reduce smoothness to allow micro-motion
                 w_vel: float = 0.3,      # velocity loss
                 w_fft: float = 0.2,      # high-frequency spectrum loss
                 w_res: float = 0.5,      # residual-based weighting
                 **kwargs):
        super(FMLossFFT, self).__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.w_fm = w_fm
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran
        self.w_smo = w_smo
        self.w_vel = w_vel
        self.w_fft = w_fft
        self.w_res = w_res

    def forward(self, gt_emotions_list, gt_3dmms_list, fm_outputs_list, **kwargs):
        total_fm_loss = 0.0
        total_rec_loss = 0.0
        total_vel_loss = 0.0
        total_fft_loss = 0.0
        total_rec_emotion_loss = 0.0
        total_rec_param_loss = 0.0

        for fm_outputs in fm_outputs_list:
            B, T_window, _ = fm_outputs['recon'].shape

            # Align GT
            gt_3dmm_window = []
            gt_emotion_window = []
            for b in range(B):
                gt_3dmm_window.append(gt_3dmms_list[b][:T_window].to(fm_outputs['recon'].device))
                gt_emotion_window.append(gt_emotions_list[b][:T_window].to(fm_outputs['listener_emotion'].device))
            gt_3dmm_window = torch.stack(gt_3dmm_window, dim=0)
            gt_emotion_window = torch.stack(gt_emotion_window, dim=0)

            pred_3dmm = fm_outputs["recon"]
            pred_emotion = fm_outputs["listener_emotion"]

            # Flow Matching loss
            fm_loss = F.mse_loss(fm_outputs["v_pred"], fm_outputs["v_true"])

            # Expression loss with RESIDUAL-BASED WEIGHTING
            pred_expr = pred_3dmm[..., :52]
            gt_expr = gt_3dmm_window[..., :52]
            residual = (pred_expr - gt_expr).abs()
            gt_mag = gt_expr.abs() + 1e-6
            rel_error = residual / gt_mag
            weights = self.w_exp * (1.0 + self.w_res * rel_error)
            weights = torch.clamp(weights, min=self.w_exp, max=self.w_exp * 5.0)
            expr_loss = (self.mse(pred_expr, gt_expr) * weights).mean()

            # Rotation & Translation
            rot_loss = self.w_rot * F.mse_loss(pred_3dmm[..., 52:55], gt_3dmm_window[..., 52:55])
            tran_loss = self.w_tran * F.mse_loss(pred_3dmm[..., 55:], gt_3dmm_window[..., 55:])
            smooth_rot = self.w_smo * temporal_smooth_loss_2(pred_3dmm[..., 52:55])
            smooth_tran = self.w_smo * temporal_smooth_loss_2(pred_3dmm[..., 55:])
            rec_param_loss = expr_loss + rot_loss + tran_loss + smooth_rot + smooth_tran

            # Velocity Loss
            if T_window > 1:
                vel_pred = pred_expr[:, 1:] - pred_expr[:, :-1]
                vel_gt = gt_expr[:, 1:] - gt_expr[:, :-1]
                vel_loss = self.w_vel * F.l1_loss(vel_pred, vel_gt)
            else:
                vel_loss = torch.tensor(0.0, device=pred_3dmm.device)

            # FFT High-Frequency Loss
            fft_loss = fft_spectrum_loss(pred_expr, gt_expr, weight=self.w_fft)

            # Emotion loss
            rec_emotion_loss = self.w_emo * F.mse_loss(pred_emotion, gt_emotion_window)

            rec_loss = rec_param_loss + rec_emotion_loss + vel_loss + fft_loss

            total_fm_loss += fm_loss
            total_rec_loss += rec_loss
            total_vel_loss += vel_loss
            total_fft_loss += fft_loss
            total_rec_emotion_loss += rec_emotion_loss
            total_rec_param_loss += rec_param_loss

        n = max(1, len(fm_outputs_list))
        total_loss = (self.w_fm * total_fm_loss + total_rec_loss) / n

        # # 打印所有分项损失
        # print("FMLossFFT components -- total: {:.6f}, fm: {:.6f}, rec: {:.6f} (emo: {:.6f}, param: {:.6f}), vel: {:.6f}, fft: {:.6f}".format(
        #     total_loss.item(),
        #     (total_fm_loss / n).item(),
        #     (total_rec_loss / n).item(),
        #     (total_rec_emotion_loss / n).item(),
        #     (total_rec_param_loss / n).item(),
        #     (total_vel_loss / n).item(),
        #     (total_fft_loss / n).item()
        # ))


        return {
            "loss": total_loss,
            "loss_rec": total_rec_loss / n,
            "loss_fm": total_fm_loss / n,
            "loss_vel": total_vel_loss / n,
            "loss_fft": total_fft_loss / n,
            "loss_emo": total_rec_emotion_loss / n,
            "loss_param": total_rec_param_loss / n,
        }

    def __repr__(self):
        return f"FMLossFFT(w_fm={self.w_fm}, w_exp={self.w_exp}, w_vel={self.w_vel}, w_fft={self.w_fft})"
