import math
from pathlib import Path
from typing import List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_metrics
from framework.utils.losses import div_loss_flow
from framework.utils.util import AverageMeter, from_pretrained_checkpoint

os.environ["NUMEXPR_MAX_THREADS"] = '16'
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):

        self.model_cfg = model
        self.criterion_cfg = criterion
        self.resumed_training = resumed_training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.val_interval = kwargs.pop("val_interval", 10)
        self.lr = kwargs.pop('lr')
        self.optim_cfg = kwargs.pop("optim")
        self.epochs = kwargs.pop("epochs")
        self.gpu_ids = kwargs.pop("gpu_ids")
        self.j = kwargs.pop("j")
        self.max_seq_len = kwargs.pop("max_seq_len")
        self.window_size = kwargs.pop("window_size")
        self.validation_window_size = int(kwargs.pop("validation_window_size", 8))
        self.task = kwargs.pop("task")
        self.kwargs = kwargs

        self.cur_epoch = 0

    def get_ckpt_path(self, model, runid="current_runid", epoch=None, best=False, last=False, save=True):
        ckpt_dir = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
        run_id = Path(self.kwargs.get(runid))
        ckpt_dir = str(ckpt_dir / run_id / model.get_model_name())
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = None
        if epoch is not None:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth")
        if best:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
            # 如果不存在best，则使用last
            if not os.path.exists(ckpt_path) and not save:
                ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
        if last:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
        assert ckpt_path is not None, "No checkpoint path is provided."
        return ckpt_path

    def set_data_module(self, data_module):
        self.data_module = data_module

    def data_resample(self, speaker_audio_clips, speaker_video_clips, listener_emotion_clips,
                      listener_3dmm_clips, speaker_seq_lengths, listener_seq_lengths):
        speaker_audios = [audio[:L] for audio, L in zip(speaker_audio_clips, speaker_seq_lengths)]
        speaker_videos = [video[:L] for video, L in zip(speaker_video_clips, speaker_seq_lengths)]
        listener_emotions = [emo[:L] for emo, L in zip(listener_emotion_clips, listener_seq_lengths)]
        listener_3dmm = [param[:L] for param, L in zip(listener_3dmm_clips, listener_seq_lengths)]
        return speaker_audios, speaker_videos, listener_emotions, listener_3dmm

    def fit(self):
        """
        # relative directory
        root_dir = save/${trainer.task_name}/${data.data_name}/${folder_name}
        # absolute directory
        saving_dir = Path(hydra.utils.to_absolute_path(root_dir))
        # get saving path
        saving_path = str(saving_dir / ...)
        """
        stage = "fit"

        logger.info("Loading data module")
        self.train_loader, self.val_loader = (
            self.data_module.get_dataloader(stage=stage))
        logger.info("Data module loaded")
        if len(self.train_loader) == 0 or len(self.val_loader) == 0:
            raise RuntimeError("Training and validation loaders must both contain samples")

        logger.info("Loading criterion")
        self.criterion = instantiate(self.criterion_cfg)
        self.criterion.to(self.device)
        logger.info("Criterion loaded")

        # criterion拥有__repr__方法可以返回名称
        logger.info(f"Using criterion: {repr(self.criterion)}")

        self.main()

    def main(self):
        model = instantiate(self.model_cfg, _recursive_=False)
        model.to(self.device)
        optimizer = instantiate(self.optim_cfg, lr=self.lr, params=model.parameters())

        # 如果模型存在get_version方法，打印版本信息
        if hasattr(model, 'get_version'):
            version_info = model.get_version()
            logger.info(f"Model version: {version_info}")

        if self.resumed_training:
            checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", last=True, save=False)
            from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            lowest_val_loss, start_epoch = from_pretrained_checkpoint(checkpoint_path, model, self.device)
            logger.info(f"Resume training from epoch {start_epoch}")
        else:
            start_epoch = 0
            lowest_val_loss = float('inf')
        print(f"Best validation loss: {lowest_val_loss}")

        for epoch in range(start_epoch, self.epochs):
            self.cur_epoch = epoch + 1
            train_loss = self.train(model, optimizer)
            logger.info("Epoch: {}  train_loss: {:.5f}"
                  .format(epoch + 1, train_loss))

            if (epoch + 1) % self.val_interval == 0 or epoch + 1 == self.epochs:
                val_loss = self.val(model)
                logger.info("Epoch: {}  val_loss: {:.5f}"
                      .format(epoch + 1,  val_loss))

                checkpoint = {
                    'epoch': epoch + 1,
                    'best_loss': lowest_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    ckpt_path = self.get_ckpt_path(model, best=True)
                    logger.info(f"Saving best checkpoint, val_loss: {lowest_val_loss:.5f}, ckpt_path: {ckpt_path}")
                    checkpoint['best_loss'] = lowest_val_loss
                    torch.save(checkpoint, ckpt_path)

                # Temp 保存逻辑，20 epoch 之后，每10个 epoch 保存一次
                save = True

                if save:
                    ckpt_path = self.get_ckpt_path(model, epoch=(epoch + 1))
                    torch.save(checkpoint, ckpt_path)

            torch.save({
                'epoch': epoch + 1,
                'best_loss': lowest_val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, self.get_ckpt_path(model, last=True))

    # Train
    def train(self, model, optimizer):
        losses = AverageMeter()

        model.train()
        for batch_idx, (_,
                        _,
                        _,
                        _,
                        _,
                        listener_emotion,
                        listener_3dmm,
                        _,
                        _
                        ) in enumerate(tqdm(self.train_loader)):
            optimizer.zero_grad()

            # print("listener_emotion shape:", listener_emotion.shape)
            # print("listener_3dmm shape:", listener_3dmm.shape)

            listener_emotion = torch.stack(listener_emotion, dim=0).to(self.device)
            listener_3dmm = torch.stack(listener_3dmm, dim=0).to(self.device)

            # === Forward pass (训练模式，Flow Matching 主 loss) ===
            outputs = model(
                emotion=listener_emotion,
                params=listener_3dmm,
            )

            loss_dict = self.criterion(listener_emotion, listener_3dmm, outputs)
            loss = loss_dict['total']

            # === 总 loss ===
            total_loss = loss

            # === 记录指标 ===
            batch_size = len(listener_3dmm)
            losses.update(total_loss.item(), batch_size)

            # === backward ===
            total_loss.backward()
            optimizer.step()

        return losses.avg


    # Validation
    def val(self, model):
        losses = AverageMeter()

        model.eval()
        model.reset_window_size(self.validation_window_size)

        for batch_idx, (_,
                        _,
                        _,
                        _,
                        _,
                        listener_emotion,
                        listener_3dmm,
                        _,
                        _
                        ) in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():

                listener_emotion = torch.stack(listener_emotion, dim=0).to(self.device)
                listener_3dmm = torch.stack(listener_3dmm, dim=0).to(self.device)

                outputs = model(
                    emotion=listener_emotion,
                    params=listener_3dmm,
                )

                loss_dict = self.criterion(listener_emotion, listener_3dmm, outputs)

                # "loss": total_loss,

                loss = loss_dict['total']
                total_loss = loss

                batch_size = len(listener_3dmm)
                losses.update(total_loss.item(), batch_size)

        model.reset_window_size(self.window_size)
        return losses.avg

    def pad_to(self, seq: torch.Tensor, length: int) -> torch.Tensor:
        L = seq.shape[0]
        if L < length:
            pad_shape = (length - L, *seq.shape[1:])
            return torch.cat([seq, seq.new_zeros(pad_shape)], dim=0)
        return seq
