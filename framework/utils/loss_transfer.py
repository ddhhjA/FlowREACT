import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_smooth(x):
    return torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))


class Emotion3DMMLoss(nn.Module):
    def __init__(self,
                 w_rec: float = 1.0,
                 w_latent: float = 0.5,
                 w_smo: float = 0.1,
                 **kwargs):
        super(Emotion3DMMLoss, self).__init__()
        self.w_rec = w_rec
        self.w_latent = w_latent
        self.w_smo = w_smo


    def forward(self, gt_emotion, gt_3dmm, outputs, **kwargs):
        # emotion, params: (B, T, D)
        pred_3dmm = outputs["pred_3dmm"]
        pred_emotion = outputs["pred_emotion"]
        z_e = outputs["z_e"]
        z_p = outputs["z_p"]

        loss = {}

        loss["rec_3dmm"] = F.l1_loss(pred_3dmm, gt_3dmm)
        loss["rec_emotion"] = F.l1_loss(pred_emotion, gt_emotion)

        loss["latent"] = F.mse_loss(z_e, z_p)

        loss["smooth"] = temporal_smooth(z_e) + temporal_smooth(z_p)

        loss["total"] = (
                self.w_rec * (loss["rec_3dmm"] + loss["rec_emotion"]) +
                self.w_latent * loss["latent"] +
                self.w_smo * loss["smooth"]
        )

        return loss

    def __repr__(self):
        return "Emotion3DMMLoss"