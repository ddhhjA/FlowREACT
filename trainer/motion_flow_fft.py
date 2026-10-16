import math
import csv
import gc
from pathlib import Path
from typing import List
import cv2
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from framework.modules.post_processor import Processor
from framework.motion_transvae.TransformerVAE import Intra3DMMEncoder
from framework.utils.compute_metrics import compute_metrics
from framework.utils.losses import div_loss_flow
from framework.utils.robustness import (
    apply_input_perturbation,
    clone_tensor_list,
    compute_audio_noise_audit,
    compute_input_audit,
    compute_expression_activity,
    mismatch_audio_from_previous_batch,
    robustness_condition_name,
    save_robustness_debug_visualization,
    validate_robustness_mode,
)
from framework.utils.util import AverageMeter, from_pretrained_checkpoint
from framework.utils.model_mode import temporary_eval
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchinfo import summary
import time

os.environ["NUMEXPR_MAX_THREADS"] = '16'
logger = logging.getLogger(__name__)


class _FlowTrainingForwardWrapper(nn.Module):
    """Tensor-only adapter used by fvcore's JIT tracer."""

    def __init__(self, model, use_listener_gt, history_weight=1.0):
        super().__init__()
        self.model = model
        self.use_listener_gt = use_listener_gt
        self.history_weight = history_weight

    def forward(self, speaker_3dmm, speaker_audio, listener_3dmm, motion_lengths):
        speaker_3dmm_list = torch.unbind(speaker_3dmm, dim=0)
        speaker_audio_list = torch.unbind(speaker_audio, dim=0)
        if self.use_listener_gt:
            listener_3dmm_list = torch.unbind(listener_3dmm, dim=0)
            outputs = self.model(
                speaker_3dmm_list,
                speaker_audio_list,
                listener_gt=listener_3dmm_list,
                motion_lengths=motion_lengths,
                history_weight=self.history_weight,
            )
        else:
            outputs = self.model(speaker_3dmm_list, speaker_audio_list,
                                 motion_lengths=motion_lengths, history_weight=self.history_weight)
        return outputs["listener_3dmm_out"], outputs["listener_emotion_out"]


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 renderer: DictConfig = None,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):

        self.renderer_cfg = renderer
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
        self.div_p = kwargs.pop("div_p")
        self.task = kwargs.pop("task")
        self.robustness_cfg = kwargs.pop("robustness", None)
        self.train_profile_cfg = kwargs.pop("train_profile", None)
        self.inference_history = kwargs.pop("inference_history", "previous_window")
        self.history_weight = float(kwargs.pop("history_weight", 1.0))
        self.history_training = kwargs.pop("history_training", None) or {}
        self.history_sample_probability = float(self.history_training.get("sample_probability", 0.0))
        self.history_warmup_epochs = int(self.history_training.get("warmup_epochs", 0))
        self.history_ramp_epochs = int(self.history_training.get("ramp_epochs", 10))
        self.init_checkpoint = kwargs.pop("init_checkpoint", None)
        if not 0.0 <= self.history_weight <= 1.0:
            raise ValueError("history_weight must be in [0, 1]")
        if not 0.0 <= self.history_sample_probability <= 1.0:
            raise ValueError("history_training.sample_probability must be in [0, 1]")
        if self.history_warmup_epochs < 0 or self.history_ramp_epochs < 0:
            raise ValueError("History warmup/ramp epochs must be nonnegative")
        if self.init_checkpoint and self.resumed_training:
            raise ValueError("Use either init_checkpoint (new fine-tuning run) or resume, not both")
        if self.history_sample_probability > 0 and (
                self.task != "online" or self.history_weight == 0 or self.inference_history == "legacy_zero"):
            raise ValueError("Sampled-history training requires online, previous_window and history_weight > 0")
        self.evaluation_cfg = kwargs.pop("evaluation", None)
        self.speaker_independent_cfg = kwargs.pop("speaker_independent", None) or {}
        self.split_protocol = self.speaker_independent_cfg.get("protocol", "speaker_independent")
        if self.split_protocol not in {"speaker_independent", "recording_disjoint"}:
            raise ValueError("Unknown isolated-split protocol: {}".format(self.split_protocol))
        self.speaker_protocol = None
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

        if self.kwargs.get("require_pretrained_emotion_transfer", False):
            checkpoint = self.model_cfg.get("emotion_transfer_ckpt")
            if not checkpoint:
                raise ValueError("Training requires an explicit pretrained emotion_transfer_ckpt")
            if not Path(hydra.utils.to_absolute_path(checkpoint)).is_file():
                raise FileNotFoundError(f"Emotion converter checkpoint not found: {checkpoint}")

        logger.info("Loading data module")
        self.train_loader, self.val_loader = (
            self.data_module.get_dataloader(stage=stage))
        logger.info("Data module loaded")
        if len(self.train_loader) == 0 or len(self.val_loader) == 0:
            raise RuntimeError("Training and validation loaders must both contain samples")
        if self.speaker_independent_cfg.get("enabled", False):
            from framework.utils.speaker_split import verify_converter
            audit = getattr(self.train_loader.dataset, "identity_protocol", None)
            dev_audit = getattr(self.val_loader.dataset, "identity_protocol", None)
            if (not audit or not dev_audit or audit["split_id"] != dev_audit["split_id"]
                    or audit["protocol"] != self.split_protocol or dev_audit["protocol"] != self.split_protocol):
                raise ValueError("Isolated experiment requires matching audited train/development manifests and protocol")
            if self.init_checkpoint:
                raise ValueError("Isolated training starts fresh, not from an exposed FlowREACT checkpoint")
            converter_path = self.model_cfg.get("emotion_transfer_ckpt")
            if not converter_path:
                raise ValueError("Train the emotion converter on the isolated training partition first")
            converter = verify_converter(
                hydra.utils.to_absolute_path(str(converter_path)),
                Path(hydra.utils.to_absolute_path(self.train_loader.dataset.pair_manifest)).parent,
            )
            self.speaker_protocol = {
                "protocol": audit["protocol"],
                "split_id": audit["split_id"], "training_run_id": str(self.kwargs.get("current_runid")),
                "training_seed": int(self.speaker_independent_cfg["training_seed"]),
                "converter_sha256": converter["checkpoint_sha256"],
                "counts": audit["counts"],
            }
            if audit["protocol"] == "speaker_independent":
                self.speaker_protocol["unseen_speaker_ids"] = audit["unseen_speaker_ids"]
            logger.info("Audited isolated training: %s", self.speaker_protocol)
        logger.info(
            "Training dataset=%s (%d samples), validation dataset=%s (%d samples), resume=%s",
            getattr(self.train_loader.dataset, "split", "train"), len(self.train_loader.dataset),
            getattr(self.val_loader.dataset, "split", "val"), len(self.val_loader.dataset),
            self.resumed_training,
        )

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
            if self.speaker_protocol:
                proof = torch.load(checkpoint_path, map_location="cpu").get("speaker_protocol")
                if (not proof or proof["split_id"] != self.speaker_protocol["split_id"]
                        or proof.get("protocol", "speaker_independent") != self.split_protocol
                        or proof.get("training_seed") != self.speaker_protocol["training_seed"]
                        or proof.get("converter_sha256") != self.speaker_protocol["converter_sha256"]):
                    raise ValueError("Resume checkpoint is not from this isolated split")
                self.speaker_protocol["training_run_id"] = proof["training_run_id"]
            from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            lowest_val_loss, start_epoch = from_pretrained_checkpoint(checkpoint_path, model, self.device)
            logger.info(f"Resume training from epoch {start_epoch}")
        else:
            start_epoch = 0
            lowest_val_loss = float('inf')
            if self.init_checkpoint:
                source = Path(hydra.utils.to_absolute_path(str(self.init_checkpoint))).resolve()
                destination = Path(self.get_ckpt_path(model, last=True)).resolve().parent
                if source.parent == destination:
                    raise ValueError("Fine-tuning requires a new run_id to preserve the source checkpoint")
                from_pretrained_checkpoint(str(source), model, self.device)
                logger.info("Fine-tuning weights from %s; optimizer, epoch and best loss start fresh", source)
        print(f"Best validation loss: {lowest_val_loss}")

        for epoch in range(start_epoch, self.epochs):
            self.cur_epoch = epoch + 1
            train_loss, fm_loss, rec_loss, rec_emo_loss, rec_param_loss, div_loss = self.train(model, optimizer)
            logger.info("Epoch: {}  train_loss: {:.5f} fm_loss: {:.5f}  rec_all_loss: {:.5f}  rec_emo_loss: {:.5f}  "
                        "rec_parma_loss: {:.5f}  div_loss: {:.5f}"
                  .format(epoch + 1, train_loss, fm_loss, rec_loss, rec_emo_loss, rec_param_loss, div_loss))

            if (epoch + 1) % self.val_interval == 0 or epoch + 1 == self.epochs:
                val_loss, fm_loss, rec_loss, rec_emo_loss, rec_param_loss = self.val(model)
                logger.info("Epoch: {}  val_loss: {:.5f}  fm_loss: {:.5f}  rec_all_loss: {:.5f}  rec_emo_loss: {:.5f}  "
                            "rec_param_loss: {:.5f}"
                      .format(epoch + 1,  val_loss,fm_loss, rec_loss, rec_emo_loss, rec_param_loss))

                checkpoint = {
                    'epoch': epoch + 1,
                    'best_loss': lowest_val_loss,
                    'state_dict': model.state_dict(),
                    'speaker_protocol': self.speaker_protocol,
                    'optimizer': optimizer.state_dict(),
                    'history_settings': self._history_settings(),
                }

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    ckpt_path = self.get_ckpt_path(model, best=True)
                    logger.info(f"Saving best checkpoint, val_loss: {lowest_val_loss:.5f}, ckpt_path: {ckpt_path}")
                    checkpoint['best_loss'] = lowest_val_loss
                    torch.save(checkpoint, ckpt_path)

                # Temp 保存逻辑，20 epoch 之后，每10个 epoch 保存一次
                save = True
                if epoch + 1 > 20:
                    if (epoch + 1) % 10 != 0:
                        save = False

                if save:
                    ckpt_path = self.get_ckpt_path(model, epoch=(epoch + 1))
                    torch.save(checkpoint, ckpt_path)

            # Preserve the latest optimizer/model even between validation epochs.
            torch.save({
                'epoch': epoch + 1,
                'best_loss': lowest_val_loss,
                'state_dict': model.state_dict(),
                'speaker_protocol': self.speaker_protocol,
                'optimizer': optimizer.state_dict(),
                'history_settings': self._history_settings(),
            }, self.get_ckpt_path(model, last=True))

    def _history_settings(self):
        return {
            "history_weight": self.history_weight,
            "inference_history": self.inference_history,
            "sample_probability": self.history_sample_probability,
            "warmup_epochs": self.history_warmup_epochs,
            "ramp_epochs": self.history_ramp_epochs,
        }

    def _history_probability(self):
        elapsed = max(0, self.cur_epoch - self.history_warmup_epochs)
        progress = min(1.0, elapsed / max(1, self.history_ramp_epochs))
        return self.history_sample_probability * progress

    def _training_forwards(self, model, speaker_3dmm, speaker_audio, listener_3dmm, lengths):
        probability = self._history_probability()
        sampled = None
        if probability > 0:
            # Reuse the diversity rollout as detached, deployment-like history.
            with torch.no_grad(), temporary_eval(model):
                sampled = model(speaker_3dmm, speaker_audio, motion_lengths=lengths,
                                history_mode="previous_window", history_weight=self.history_weight)
        supervised = model(
            speaker_3dmm, speaker_audio, listener_gt=listener_3dmm, motion_lengths=lengths,
            history_weight=self.history_weight, sampled_history=sampled,
            sampled_history_probability=probability,
        )
        if sampled is None:
            with torch.no_grad():
                sampled = model(speaker_3dmm, speaker_audio, motion_lengths=lengths,
                                history_weight=self.history_weight)
        return supervised, sampled

    # Train
    def train(self, model, optimizer):
        losses = AverageMeter()
        fm_losses = AverageMeter()
        rec_losses = AverageMeter()
        div_losses = AverageMeter()
        rec_emo_losses = AverageMeter()
        rec_param_losses = AverageMeter()

        model.train()
        logger.info("History training: weight=%.3f, sampled-history probability=%.3f",
                    self.history_weight, self._history_probability())
        for batch_idx, (speaker_audio_clip,
                        speaker_video_clip,
                        _, speaker_3dmm, _,
                        listener_emotion,
                        listener_3dmm,
                        speaker_clip_length,
                        listener_clip_length
                        ) in enumerate(tqdm(self.train_loader)):
            optimizer.zero_grad()

            outputs, outputs_1 = self._training_forwards(
                model, speaker_3dmm, speaker_audio_clip, listener_3dmm, speaker_clip_length,
            )

            d_loss = (div_loss_flow(outputs_1["listener_3dmm_out"], outputs["listener_3dmm_out"]) +
                      div_loss_flow(outputs_1["listener_emotion_out"], outputs["listener_emotion_out"]))

            loss_dict = self.criterion(listener_emotion, listener_3dmm, outputs["fm_outputs"], cur_stage=(self.cur_epoch / self.epochs))
            loss = loss_dict['loss']

            # === 总 loss ===
            total_loss = loss + self.div_p * d_loss

            # "loss": total_loss,
            # "loss_rec": rec_loss,
            # "loss_emo": rec_emotion_loss,
            # "loss_param": rec_param_loss,
            # "loss_fm": fm_loss
            loss_fm = loss_dict.get("loss_fm", 0.0)
            loss_rec = loss_dict.get("loss_rec", 0.0)
            loss_rec_emo = loss_dict.get("loss_emo", 0.0)
            loss_rec_param = loss_dict.get("loss_param", 0.0)


            # print(f"batch {batch_idx}: total_loss={total_loss.item():.4f}, loss_fm={loss_fm.item():.4f}, ")
            # print(f"             rec_loss={loss_rec.item():.4f}, loss_rec_emo={loss_rec_emo.item():.4f}, ")
            # print(f"             loss_rec_param={loss_rec_param.item():.4f}, div_loss={d_loss.item():.4f}")


            # === 记录指标 ===
            batch_size = len(speaker_3dmm)
            losses.update(total_loss.item(), batch_size)
            fm_losses.update(loss_fm.item(), batch_size)
            rec_losses.update(loss_rec.item(), batch_size)
            rec_emo_losses.update(loss_rec_emo.item(), batch_size)
            rec_param_losses.update(loss_rec_param.item(), batch_size)
            div_losses.update(d_loss.item(), batch_size)

            # === backward ===
            total_loss.backward()
            optimizer.step()

        return losses.avg, fm_losses.avg, rec_losses.avg, rec_emo_losses.avg, rec_param_losses.avg, div_losses.avg

    # Validation
    def val(self, model):
        losses = AverageMeter()
        fm_losses = AverageMeter()
        rec_losses = AverageMeter()
        rec_emo_losses = AverageMeter()
        rec_param_losses = AverageMeter()

        model.eval()
        model.reset_window_size(self.validation_window_size)

        for batch_idx, (speaker_audio_clip,
                        speaker_video_clip,
                        _, speaker_3dmm, _,
                        listener_emotion,
                        listener_3dmm,
                        speaker_clip_length,
                        listener_clip_length) in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():
                # forward (val 不需要 diversity loss)
                outputs = model(
                    speaker_3dmm, speaker_audio_clip,
                    listener_gt=listener_3dmm,
                    motion_lengths=speaker_clip_length,
                    history_weight=self.history_weight,
                )

                loss_dict = self.criterion(listener_emotion, listener_3dmm, outputs["fm_outputs"], cur_stage=1.0)

                # "loss": total_loss,
                # "loss_rec": rec_loss,
                # "loss_emo": rec_emotion_loss,
                # "loss_param": rec_param_loss,
                # "loss_fm": fm_loss
                loss = loss_dict['loss']
                loss_fm = loss_dict.get("loss_fm", 0.0)
                loss_rec = loss_dict.get("loss_rec", 0.0)
                loss_rec_emo = loss_dict.get("loss_emo", 0.0)
                loss_rec_param = loss_dict.get("loss_param", 0.0)

                batch_size = len(speaker_3dmm)
                losses.update(loss.item(), batch_size)
                fm_losses.update(loss_fm.item(), batch_size)
                rec_losses.update(loss_rec.item(), batch_size)
                rec_emo_losses.update(loss_rec_emo.item(), batch_size)
                rec_param_losses.update(loss_rec_param.item(), batch_size)

        model.reset_window_size(self.window_size)
        return losses.avg, fm_losses.avg, rec_losses.avg, rec_emo_losses.avg, rec_param_losses.avg

    def pad_to(self, seq: torch.Tensor, length: int) -> torch.Tensor:
        L = seq.shape[0]
        if L < length:
            pad_shape = (length - L, *seq.shape[1:])
            return torch.cat([seq, seq.new_zeros(pad_shape)], dim=0)
        return seq

    def test(self):
        stage = "test"
        speaker_experiment = bool(self.speaker_independent_cfg.get("enabled", False))
        if speaker_experiment and "file_system" in torch.multiprocessing.get_all_sharing_strategies():
            torch.multiprocessing.set_sharing_strategy("file_system")
        data_clamp = self.kwargs.pop("data_clamp")
        evaluation_cfg = self.evaluation_cfg or {}
        evaluation_protocol = evaluation_cfg.get("protocol", "paired")
        if evaluation_protocol != "paired":
            raise ValueError(f"Unknown evaluation protocol: {evaluation_protocol}")
        if self.inference_history not in {"previous_window", "legacy_zero"}:
            raise ValueError(f"Unknown inference_history: {self.inference_history}")

        robustness_cfg = self.robustness_cfg or {}
        robustness_enabled = bool(robustness_cfg.get("enabled", False))
        robustness_mode = validate_robustness_mode(
            robustness_cfg.get("mode", "clean") if robustness_enabled else "clean"
        )
        audio_snr_db = float(robustness_cfg.get("audio_snr_db", 20.0))
        expression_dims = int(robustness_cfg.get("expression_dims", 52))
        metric_threads = int(robustness_cfg.get("metric_threads", 4))
        max_robustness_samples = int(robustness_cfg.get("max_samples", 0))
        robustness_seed = int(robustness_cfg.get("seed", 1234))
        save_debug_visuals = bool(robustness_cfg.get("save_debug_visuals", False))
        debug_max_frames = int(robustness_cfg.get("debug_max_frames", 128))
        debug_window_index = int(robustness_cfg.get("debug_window_index", 1))
        model_ablation_modes = {
            "speaker_condition_zero",
            "speaker_feature_zero",
            "history_zero",
            "flow_noise_zero",
            "flow_latent_zero",
            "periodic_zero",
            "hard_zero",
        }
        model_ablation_enabled = robustness_mode in model_ablation_modes
        condition_name = robustness_condition_name(robustness_mode, audio_snr_db)
        if speaker_experiment:
            if robustness_enabled:
                raise ValueError("Isolated experiment uses unperturbed REACT2025 inputs")
            condition_name = str(self.speaker_independent_cfg.get("evaluation_group"))
            groups = {"heldout"} if self.split_protocol == "recording_disjoint" else {"seen", "unseen"}
            if condition_name not in groups:
                raise ValueError("Evaluation group does not match the isolated-split protocol")
            result_dir = Path(hydra.utils.to_absolute_path(
                self.speaker_independent_cfg["output_dir"])) / condition_name / f"seed_{robustness_seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
        elif robustness_enabled:
            output_root = hydra.utils.to_absolute_path(
                robustness_cfg.get("output_dir", "results/robustness")
            )
            result_dir = Path(output_root) / condition_name / f"seed_{robustness_seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
        else:
            result_dir = Path.cwd()
        logger.info(
            "Robustness evaluation: enabled=%s, condition=%s, output=%s",
            robustness_enabled,
            condition_name,
            result_dir,
        )

        model = instantiate(self.model_cfg, _recursive_=False)
        requested_checkpoint = self.speaker_independent_cfg.get("checkpoint") if speaker_experiment else None
        checkpoint_path = (hydra.utils.to_absolute_path(str(requested_checkpoint)) if requested_checkpoint
                           else self.get_ckpt_path(model, runid="resume_runid", best=True, save=False))
        # checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", epoch=20)
        # checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", epoch=100)
        from_pretrained_checkpoint(checkpoint_path, model, self.device)
        model.eval()

        # Rendering is disabled for robustness runs so every sample reaches metric evaluation.
        render_enabled = (bool(self.renderer_cfg.get("do_render", False))
                          and not robustness_enabled and not speaker_experiment)
        renderer = None
        if render_enabled:
            renderer = instantiate(
                self.renderer_cfg,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        logger.info("Loading test data module")
        test_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Test data module loaded")

        evaluation_metadata = {"protocol": evaluation_protocol}
        if speaker_experiment:
            from framework.utils.speaker_split import verify_converter
            dataset = test_loader.dataset
            audit = getattr(dataset, "identity_protocol", None)
            if not audit or audit["partition"] != condition_name or audit["protocol"] != self.split_protocol:
                raise ValueError("Evaluation group must match its audited sample manifest")
            proof = torch.load(checkpoint_path, map_location="cpu").get("speaker_protocol")
            if (not proof or proof["split_id"] != audit["split_id"]
                    or proof.get("protocol", "speaker_independent") != self.split_protocol):
                raise ValueError("FlowREACT checkpoint was not trained on this isolated split")
            if proof.get("training_seed") != robustness_seed:
                raise ValueError("Evaluation seed must match the independent training seed")
            converter = verify_converter(
                hydra.utils.to_absolute_path(str(self.model_cfg.get("emotion_transfer_ckpt"))),
                Path(hydra.utils.to_absolute_path(dataset.pair_manifest)).parent,
            )
            if converter["checkpoint_sha256"] != proof["converter_sha256"]:
                raise ValueError("Converter differs from the converter used for FlowREACT training")
            if (not isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)
                    or test_loader.drop_last or max_robustness_samples):
                raise ValueError("Isolated evaluation requires the complete ordered manifest")
            if self.max_seq_len % self.window_size:
                raise ValueError("max_seq_len must be divisible by window_size")
            evaluation_metadata.update({
                "protocol": audit["protocol"], "split_id": audit["split_id"],
                "group": condition_name, "counts": audit["counts"][condition_name],
                "training_run_id": proof["training_run_id"],
                "training_seed": proof["training_seed"],
                "converter_sha256": proof["converter_sha256"],
                "gt_seed": dataset.gt_seed, "input_perturbation": "none",
                "listener_identity_disjoint": audit.get("listener_identity_disjoint"),
                "speaker_identity_disjoint": (True if self.split_protocol == "speaker_independent" else None),
                "scope": audit["scope"],
            })
            logger.info("Audited isolated evaluation: %s", evaluation_metadata)
        logger.info("Loading post processor")
        post_processor = Processor(config_name=self.kwargs.pop("post_config_name"),
                                   clip_len_test=self.kwargs.pop("post_clip_length"),
                                   device=self.device)
        logger.info("Post processor loaded")
        evaluation_metadata.update({
            "history_weight": self.history_weight,
            "window_size": self.window_size,
            "max_seq_len": self.max_seq_len,
            "loader_batch_size": test_loader.batch_size,
            "checkpoint": str(checkpoint_path),
        })
        logger.info("Inference listener history: %s, weight=%.3f, window=%d, max_seq_len=%d, "
                    "loader_batch_size=%s, checkpoint=%s", self.inference_history, self.history_weight,
                    self.window_size, self.max_seq_len, test_loader.batch_size, checkpoint_path)

        speaker_emotions_input_all = []
        listener_3dmm_preds_lists_all = []
        listener_emotion_preds_lists_all = []
        listener_3dmm_GTs_all = []
        listener_emotion_GTs_all = []
        max_seq_len = self.max_seq_len
        num_preds = 10
        render_count = 0
        previous_clean_audio_clips = None
        expression_activity_all = []
        input_audit = None
        debug_saved = False
        skipped_mismatch_samples = 0
        evaluated_samples = 0

        for batch_idx, (speaker_audio_clips, speaker_video_clips, speaker_emotion_clips, speaker_3dmm_clips,
                        listener_video_clips, listener_emotions, listener_3dmms, speaker_clip_lengths,
                        listener_clip_lengths) in enumerate(tqdm(test_loader)):

            batch_size = len(speaker_clip_lengths)
            required_inputs = {
                "speaker_audio_clips": speaker_audio_clips,
                "speaker_3dmm_clips": speaker_3dmm_clips,
                "speaker_emotion_clips": speaker_emotion_clips,
                "listener_emotions": listener_emotions,
            }
            if batch_size == 0:
                raise ValueError("Test batch contains no speaker samples")
            for name, clips in required_inputs.items():
                if len(clips) != batch_size:
                    raise ValueError(
                        f"Test batch {batch_idx}: {name} has {len(clips)} samples; "
                        f"expected {batch_size}. Enable the required feature/label loading."
                    )
            if any(int(length) <= 0 for length in speaker_clip_lengths):
                raise ValueError(f"Test batch {batch_idx}: speaker clip lengths must be positive")
            if render_enabled and len(listener_video_clips) != batch_size:
                raise ValueError("Rendering requires load_video_l=true and one video per sample")

            clean_audio_clips = clone_tensor_list(speaker_audio_clips)
            if robustness_mode == "audio_mismatch":
                if previous_clean_audio_clips is None:
                    previous_clean_audio_clips = clean_audio_clips
                    skipped_mismatch_samples += len(speaker_audio_clips)
                    continue
                speaker_audio_clips = mismatch_audio_from_previous_batch(
                    speaker_audio_clips,
                    previous_clean_audio_clips,
                )
                previous_clean_audio_clips = clean_audio_clips

            if max_robustness_samples > 0 and evaluated_samples >= max_robustness_samples:
                break

            expression_activity_all.extend(
                compute_expression_activity(clip, length, expression_dims)
                for clip, length in zip(speaker_3dmm_clips, speaker_clip_lengths)
            )
            speaker_audio_clips, speaker_3dmm_clips = apply_input_perturbation(
                speaker_audio_clips,
                speaker_3dmm_clips,
                speaker_clip_lengths,
                mode=robustness_mode,
                audio_snr_db=audio_snr_db,
                expression_dims=expression_dims,
            )
            if input_audit is None:
                input_audit = compute_input_audit(
                    speaker_audio_clips,
                    speaker_3dmm_clips,
                    expression_dims=expression_dims,
                )
                if robustness_mode == "audio_noise":
                    input_audit["audio_noise"] = compute_audio_noise_audit(
                        clean_audio_clips,
                        speaker_audio_clips,
                        speaker_clip_lengths,
                        requested_snr_db=audio_snr_db,
                    )
                logger.info("Post-perturbation input audit: %s", input_audit)
            evaluated_samples += len(speaker_audio_clips)

            listener_3dmm_preds = []
            listener_emotion_preds = []
            for i in range(num_preds):
                if i == 0:
                    if speaker_experiment:
                        # Do not retain DataLoader shared-memory handles for the entire split.
                        listener_emotion_GTs_all.extend(clone_tensor_list(gt) for gt in listener_emotions)
                        speaker_emotions_input_all.extend(clone_tensor_list(speaker_emotion_clips))
                    else:
                        listener_emotion_GTs_all.extend(listener_emotions)
                        listener_3dmm_GTs_all.extend(listener_3dmms)
                        speaker_emotions_input_all.extend(speaker_emotion_clips)

                # Videos are optional; generation depends only on the cached model inputs.
                for j, (speaker_audio_clip, speaker_3dmm_clip, speaker_clip_length) in enumerate(
                        zip(speaker_audio_clips, speaker_3dmm_clips, speaker_clip_lengths)):

                    speaker_audio_clip_list = []
                    speaker_3dmm_clip_list = []
                    motion_length_list = []

                    # split into sub-clips
                    for k in range(math.ceil(speaker_clip_length / max_seq_len)):
                        start_idx = k * max_seq_len
                        end_idx = min((k + 1) * max_seq_len, speaker_clip_length)
                        motion_length = end_idx - start_idx
                        motion_length_list.append(motion_length)
                        speaker_audio_clip_list.append(speaker_audio_clip[start_idx:end_idx])
                        speaker_3dmm_clip_list.append(speaker_3dmm_clip[start_idx:end_idx])

                    if self.model_cfg.task == 'online':
                        speaker_audio_clip_list[-1] = self.pad_to(speaker_audio_clip_list[-1], max_seq_len)
                        speaker_3dmm_clip_list[-1] = self.pad_to(speaker_3dmm_clip_list[-1], max_seq_len)
                    speaker_audio_clip_inputs = speaker_audio_clip_list  # List: [tensor([l, d_audio]), ...]
                    speaker_3dmm_clip_inputs = speaker_3dmm_clip_list  # List: [tensor([l, 3, 224, 224]), ...]

                    with torch.no_grad():
                        debug_requested = save_debug_visuals and not debug_saved and i == 0 and j == 0
                        outputs = model(
                            speaker_3dmm_clip_inputs,
                            speaker_audio_clip_inputs,
                            motion_lengths=torch.tensor(motion_length_list),
                            ablation_mode=robustness_mode,
                            history_mode=self.inference_history,
                            history_weight=self.history_weight,
                            return_debug_tensors=debug_requested,
                            debug_window_index=debug_window_index,
                        )

                        if debug_requested and outputs.get("debug_tensors") is not None:
                            debug_tensors = outputs["debug_tensors"]
                            torch.save(debug_tensors, result_dir / "debug_tensors.pt")
                            debug_stats = save_robustness_debug_visualization(
                                debug_tensors,
                                result_dir / "debug_signals.png",
                                expression_dims=expression_dims,
                                max_frames=debug_max_frames,
                            )
                            with (result_dir / "debug_stats.json").open("w", encoding="utf-8") as file:
                                json.dump(debug_stats, file, indent=2, ensure_ascii=False)
                            logger.info("Saved inference debug visualization to %s", result_dir / "debug_signals.png")
                            debug_saved = True

                        if model_ablation_enabled and input_audit is not None and "model" not in input_audit:
                            input_audit["model"] = outputs.get("ablation_audit", {})
                            logger.info("Model ablation audit: %s", input_audit["model"])

                        # listener_3dmm_out = torch.cat(outputs["listener_3dmm_out"], dim=0)[:speaker_clip_length]
                        # [2, 750, 58] -> [1500, 58]
                        listener_3dmm_out = outputs["listener_3dmm_out"].reshape(-1, outputs["listener_3dmm_out"].shape[
                            -1])[:speaker_clip_length]

                        listener_3dmm_out = listener_3dmm_out.detach().cpu().unsqueeze(0)

                        # listener_emotion_out = torch.cat(outputs["listener_emotion_out"], dim=0)[:speaker_clip_length]

                        # [2, 750, 25] -> [1500, 25]
                        listener_emotion_out = outputs["listener_emotion_out"].reshape(-1, outputs[
                            "listener_emotion_out"].shape[-1])[:speaker_clip_length]

                        listener_emotion_out = listener_emotion_out.detach().cpu().unsqueeze(0)

                        if (listener_3dmm_out.shape[1] != int(speaker_clip_length)
                                or listener_emotion_out.shape[1] != int(speaker_clip_length)):
                            raise RuntimeError(
                                f"Test batch {batch_idx}, sample {j}: model output does not "
                                f"cover the {int(speaker_clip_length)} input frames"
                            )

                        if data_clamp:
                            listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])

                        # if i == 0:
                        #     # listener_3dmms = torch.stack(listener_3dmms, dim=0)
                        #     coff = {
                        #         "PRED": listener_3dmm_out[:, :300, :],
                        #         "GT": listener_3dmms
                        #     }
                        #     coff_path = os.path.join('results_videos', 'test', f"batch{str(batch_idx + 1)}_coff.pt")
                        #     os.makedirs(os.path.dirname(coff_path), exist_ok=True)
                        #     torch.save(coff, coff_path)

                        if render_enabled and i == 0:
                            val_path = os.path.join('results_videos', 'test')
                            os.makedirs(val_path, exist_ok=True)

                            # speaker_video_clip = speaker_video_clip.to(self.device)
                            # speaker_video_clip_save = speaker_video_clip[:300]  # (300, 3, 224, 224)
                            # # 保存为30fps的avi视频
                            # self.save_to_avi(speaker_video_clip_save.cpu().numpy(),
                            #                  os.path.join('results_videos', 'test',
                            #                               f"speaker_batch{str(batch_idx + 1)}.avi"),
                            #                  fps=30)
                            #
                            listener_video_clip = listener_video_clips[j][0].to(self.device)
                            if listener_video_clip.numel() == 0:
                                raise ValueError("Rendering requires nonempty listener video; set load_video_l=true")
                            # listener_video_clip_save = listener_video_clip[:300]  # (300, 3, 224, 224)
                            # # 保存为30fps的avi视频
                            # self.save_to_avi(listener_video_clip_save.cpu().numpy(),
                            #                  os.path.join('results_videos', 'test',
                            #                               f"listener_batch{str(batch_idx + 1)}.avi"),
                            #                  fps=30)


                            # 只渲染前10s（300帧）
                            listener_3dmm_out_render = listener_3dmm_out[:, :300, :]


                            perm = torch.randperm(listener_video_clip.shape[0])
                            listener_references = listener_video_clip[perm[0]]
                            assert len(listener_references.shape) == 3, \
                                "listener_references.shape should be (3, 224, 224)"

                            renderer.rendering_for_it(
                                val_path,
                                f"batch{str(batch_idx + 1)}",
                                listener_3dmm_out_render.to(self.device),
                                listener_references
                            )

                            render_count += 1

                        # 仅供测试
                        if render_count >= 100:
                            exit(0)

                    if i == 0:
                        listener_3dmm_preds.append(listener_3dmm_out)
                        listener_emotion_preds.append(listener_emotion_out)
                    else:
                        listener_3dmm_preds[j] = torch.cat(
                            (listener_3dmm_preds[j], listener_3dmm_out), dim=0)
                        listener_emotion_preds[j] = torch.cat(
                            (listener_emotion_preds[j], listener_emotion_out), dim=0)

            if len(listener_emotion_preds) != batch_size or len(listener_3dmm_preds) != batch_size:
                raise RuntimeError(f"Test batch {batch_idx}: generation did not cover every input sample")
            if batch_idx == 0:
                logger.info("First batch generated: samples=%d, candidates=%d, prediction_shape=%s",
                            batch_size, num_preds, tuple(listener_emotion_preds[0].shape))

            # listener_3dmm_preds: (num_preds, l, ...)
            listener_3dmm_preds_lists_all.extend(listener_3dmm_preds)
            # listener_emotion_preds: (num_preds, l, ...)
            listener_emotion_preds_lists_all.extend(listener_emotion_preds)

        # listener_emotion_preds_lists_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]
        # listener_emotion_GTs_all
        # List: 750 [List: [(l', 25), (l'', 25), ...], List: [(l''', 25), (l'''', 25)], ...]
        prediction_count = len(listener_emotion_preds_lists_all)
        if prediction_count == 0:
            raise RuntimeError(
                "Inference produced no predictions. Check the test split and required audio/3DMM "
                "inputs; disabled video loading must not skip generation."
            )
        if (prediction_count != len(speaker_emotions_input_all)
                or prediction_count != len(listener_emotion_GTs_all)
                or prediction_count != len(listener_3dmm_preds_lists_all)):
            raise RuntimeError("Prediction, speaker-label and listener-target sample counts do not match")
        if speaker_experiment and prediction_count != len(test_loader.dataset):
            raise RuntimeError("Evaluation did not produce a prediction for every split-manifest row")
        listener_emotion_GTs_all = post_processor.forward(
            prediction_list=listener_emotion_preds_lists_all,
            target_list=listener_emotion_GTs_all,)
        # listener_emotion_GTs_all
        # List: 750 [Tensor([num_preds, l, 25]), Tensor([num_preds, l', 25]), ...]

        tensor_output_path = result_dir / "results.pt"
        try:
            torch.save({
                'GT': listener_emotion_GTs_all,
                'PRED': listener_emotion_preds_lists_all,
                'PRED_3DMM': listener_3dmm_preds_lists_all,
                'SPEAKER_EMOTION': speaker_emotions_input_all,
                'EXPRESSION_ACTIVITY': expression_activity_all,
                'ROBUSTNESS_MODE': robustness_mode,
                'CONDITION': condition_name,
                'SEED': robustness_seed,
                'INPUT_AUDIT': input_audit,
                'CHECKPOINT': str(checkpoint_path),
                'HISTORY_MODE': self.inference_history,
                'HISTORY_WEIGHT': self.history_weight,
                'EVALUATION': evaluation_metadata,
                'SAMPLE_METADATA': getattr(test_loader.dataset, "sample_metadata", None),
            }, tensor_output_path)
            print("Successfully saved Tensor List")
        except Exception as error:
            print(f"Failed to save Tensor List: {error}")


        # 将结果列表转移到 CPU 以节省内存
        speaker_emotions_input_all = [emo.cpu() for emo in speaker_emotions_input_all]
        listener_emotion_preds_lists_all = [pred.cpu() for pred in listener_emotion_preds_lists_all]
        listener_emotion_GTs_all = [
            [item.cpu() for item in gt] if isinstance(gt, (list, tuple)) else gt.cpu()
            for gt in listener_emotion_GTs_all
        ]

        results = compute_metrics(
            speaker_emotions_input_all,
            listener_emotion_preds_lists_all,
            listener_emotion_GTs_all,
            threads=metric_threads,
        )
        logger.info(results)

        if robustness_enabled or speaker_experiment:
            metrics_payload = {
                "condition": condition_name,
                "mode": robustness_mode,
                "audio_snr_db": audio_snr_db if robustness_mode == "audio_noise" else None,
                "audio_noise_domain": (
                    "wav2vec_feature" if robustness_mode == "audio_noise" else None
                ),
                "expression_dims": expression_dims,
                "seed": robustness_seed,
                "num_samples": len(listener_emotion_preds_lists_all),
                "num_preds": num_preds,
                "skipped_mismatch_samples": skipped_mismatch_samples,
                "max_samples": max_robustness_samples,
                "checkpoint": str(checkpoint_path),
                "history_mode": self.inference_history,
                "history_weight": self.history_weight,
                "evaluation": evaluation_metadata,
                "model_ablation_enabled": model_ablation_enabled,
                "input_audit": input_audit,
                "metrics": {key: float(value) for key, value in results.items()},
            }
            with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(metrics_payload, file, indent=2, ensure_ascii=False)
            logger.info("Saved evaluation metrics to %s", result_dir / "metrics.json")

    @staticmethod
    def _next_profile_batch(data_loader, data_iterator):
        try:
            return next(data_iterator), data_iterator
        except StopIteration:
            data_iterator = iter(data_loader)
            return next(data_iterator), data_iterator

    def _profile_training_step(self, model, optimizer, criterion, batch, cur_stage):
        (speaker_audio_clip,
         _,
         _,
         speaker_3dmm,
         _,
         listener_emotion,
         listener_3dmm,
         speaker_clip_length,
         _) = batch

        optimizer.zero_grad()
        outputs, outputs_1 = self._training_forwards(
            model, speaker_3dmm, speaker_audio_clip, listener_3dmm, speaker_clip_length,
        )

        diversity_loss = (
            div_loss_flow(outputs_1["listener_3dmm_out"], outputs["listener_3dmm_out"])
            + div_loss_flow(outputs_1["listener_emotion_out"], outputs["listener_emotion_out"])
        )
        loss_dict = criterion(
            listener_emotion,
            listener_3dmm,
            outputs["fm_outputs"],
            cur_stage=cur_stage,
        )
        total_loss = loss_dict["loss"] + self.div_p * diversity_loss
        total_loss.backward()
        optimizer.step()
        return float(total_loss.detach().item()), len(speaker_3dmm)

    @staticmethod
    def _stack_profile_input(value, batch_size, device):
        if torch.is_tensor(value):
            tensor = value[:batch_size]
        else:
            tensor = torch.stack(list(value)[:batch_size], dim=0)
        return tensor.to(device)

    def _profile_training_flops(self, model, batch, requested_batch_size):
        (speaker_audio,
         _,
         _,
         speaker_3dmm,
         _,
         _,
         listener_3dmm,
         speaker_lengths,
         _) = batch

        available_batch_size = len(speaker_3dmm)
        batch_size = max(1, min(int(requested_batch_size), available_batch_size))
        speaker_audio = self._stack_profile_input(speaker_audio, batch_size, self.device)
        speaker_3dmm = self._stack_profile_input(speaker_3dmm, batch_size, self.device)
        listener_3dmm = self._stack_profile_input(listener_3dmm, batch_size, self.device)
        if torch.is_tensor(speaker_lengths):
            speaker_lengths = speaker_lengths[:batch_size].to(self.device)
        else:
            speaker_lengths = torch.as_tensor(
                [int(length) for length in list(speaker_lengths)[:batch_size]],
                device=self.device,
            )

        model.eval()
        train_wrapper = _FlowTrainingForwardWrapper(model, use_listener_gt=True, history_weight=self.history_weight).to(self.device)
        diversity_wrapper = _FlowTrainingForwardWrapper(model, use_listener_gt=False, history_weight=self.history_weight).to(self.device)
        trace_inputs = (speaker_3dmm, speaker_audio, listener_3dmm, speaker_lengths)

        train_analysis = FlopCountAnalysis(train_wrapper, trace_inputs)
        diversity_analysis = FlopCountAnalysis(diversity_wrapper, trace_inputs)
        for analysis in (train_analysis, diversity_analysis):
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            analysis.tracer_warnings("none")

        train_forward_flops = float(train_analysis.total())
        diversity_forward_flops = float(diversity_analysis.total())
        unsupported_ops = dict(train_analysis.unsupported_ops())
        for name, count in diversity_analysis.unsupported_ops().items():
            unsupported_ops[name] = unsupported_ops.get(name, 0) + count

        train_forward_per_sample = train_forward_flops / batch_size
        diversity_forward_per_sample = diversity_forward_flops / batch_size
        # Backward is conventionally approximated as twice the differentiable forward.
        training_step_per_sample = 3.0 * train_forward_per_sample + diversity_forward_per_sample
        return {
            "flops_profile_batch_size": batch_size,
            "reference_training_batch_size": available_batch_size,
            "train_forward_flops_per_sample": train_forward_per_sample,
            "diversity_forward_flops_per_sample": diversity_forward_per_sample,
            "estimated_training_step_flops_per_sample": training_step_per_sample,
            "estimated_training_step_gflops_per_sample": training_step_per_sample / 1e9,
            "estimated_training_step_tflops_per_batch": (
                training_step_per_sample * available_batch_size / 1e12
            ),
            "formula": "3 * train_forward + diversity_forward",
            "backward_assumption": "backward FLOPs ~= 2 * differentiable forward FLOPs",
            "excludes": ["criterion FLOPs", "optimizer-update FLOPs", "unsupported operators", "sampled-history selection"],
            "unsupported_ops": {str(name): int(count) for name, count in unsupported_ops.items()},
        }

    @staticmethod
    def _summarize_profile_values(runs, key):
        values = np.asarray([run[key] for run in runs], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def profile_training(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Training profiling requires a CUDA GPU")

        cfg = self.train_profile_cfg or {}
        repeats = int(cfg.get("repeats", 3))
        warmup_steps = int(cfg.get("warmup_steps", 3))
        requested_measure_steps = int(cfg.get("measure_steps", 0))
        flops_batch_size = int(cfg.get("flops_batch_size", 1))
        profile_epoch = int(cfg.get("cur_epoch", 1))
        base_seed = int(cfg.get("seed", 1234))
        output_dir = Path(hydra.utils.to_absolute_path(str(cfg.get("output_dir", "results/train_profile"))))
        output_dir.mkdir(parents=True, exist_ok=True)

        if repeats < 1:
            raise ValueError("train_profile.repeats must be at least 1")
        if warmup_steps < 1:
            raise ValueError("train_profile.warmup_steps must be at least 1 so optimizer state is allocated")

        logger.info("Loading training data for resource profiling")
        train_loader, _ = self.data_module.get_dataloader(stage="fit")
        batches_per_epoch = len(train_loader)
        if batches_per_epoch == 0:
            raise RuntimeError("The training data loader produced no batches")
        measure_steps = batches_per_epoch if requested_measure_steps == 0 else requested_measure_steps
        cur_stage = min(max(profile_epoch / self.epochs, 0.0), 1.0)
        self.cur_epoch = profile_epoch

        runs = []
        parameter_info = None
        complexity_info = None
        reference_batch = None

        for repeat_index in range(repeats):
            run_seed = base_seed + repeat_index
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            np.random.seed(run_seed)

            model = instantiate(self.model_cfg, _recursive_=False).to(self.device)
            criterion = instantiate(self.criterion_cfg).to(self.device)
            optimizer = instantiate(self.optim_cfg, lr=self.lr, params=model.parameters())
            model.train()

            if parameter_info is None:
                total_params = sum(parameter.numel() for parameter in model.parameters())
                trainable_params = sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                )
                parameter_bytes = sum(
                    parameter.numel() * parameter.element_size() for parameter in model.parameters()
                )
                buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
                parameter_info = {
                    "total_params": int(total_params),
                    "trainable_params": int(trainable_params),
                    "frozen_params": int(total_params - trainable_params),
                    "parameter_memory_mib": parameter_bytes / (1024 ** 2),
                    "buffer_memory_mib": buffer_bytes / (1024 ** 2),
                }

            data_iterator = iter(train_loader)
            for _ in range(warmup_steps):
                batch, data_iterator = self._next_profile_batch(train_loader, data_iterator)
                self._profile_training_step(model, optimizer, criterion, batch, cur_stage)
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.cuda.reset_peak_memory_stats(self.device)

            # A zero measure_steps setting means a fresh, complete epoch for each repeat.
            if requested_measure_steps == 0:
                data_iterator = iter(train_loader)

            fetch_times = []
            compute_times = []
            step_times = []
            losses = []
            measured_samples = 0

            for _ in tqdm(
                    range(measure_steps),
                    desc=f"Training profile {repeat_index + 1}/{repeats}"):
                step_start = time.perf_counter()
                batch, data_iterator = self._next_profile_batch(train_loader, data_iterator)
                fetch_end = time.perf_counter()
                if reference_batch is None:
                    reference_batch = batch

                loss_value, batch_size = self._profile_training_step(
                    model, optimizer, criterion, batch, cur_stage
                )
                torch.cuda.synchronize(self.device)
                step_end = time.perf_counter()

                fetch_times.append(fetch_end - step_start)
                compute_times.append(step_end - fetch_end)
                step_times.append(step_end - step_start)
                losses.append(loss_value)
                measured_samples += batch_size

            peak_allocated_mib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            peak_reserved_mib = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            measured_wall_seconds = float(np.sum(step_times))
            mean_step_seconds = float(np.mean(step_times))
            estimated_epoch_seconds = mean_step_seconds * batches_per_epoch
            run_result = {
                "repeat": repeat_index + 1,
                "seed": run_seed,
                "measured_steps": measure_steps,
                "measured_samples": measured_samples,
                "mean_loss": float(np.mean(losses)),
                "mean_data_time_ms": float(np.mean(fetch_times) * 1000.0),
                "mean_compute_time_ms": float(np.mean(compute_times) * 1000.0),
                "mean_step_time_ms": mean_step_seconds * 1000.0,
                "samples_per_second": measured_samples / measured_wall_seconds,
                "peak_allocated_mib": peak_allocated_mib,
                "peak_reserved_mib": peak_reserved_mib,
                "estimated_epoch_minutes": estimated_epoch_seconds / 60.0,
                "estimated_training_hours": estimated_epoch_seconds * self.epochs / 3600.0,
            }
            runs.append(run_result)
            logger.info(
                "Profile repeat %d/%d: step=%.3f ms, peak allocated=%.1f MiB, "
                "peak reserved=%.1f MiB",
                repeat_index + 1,
                repeats,
                run_result["mean_step_time_ms"],
                peak_allocated_mib,
                peak_reserved_mib,
            )

            if complexity_info is None:
                try:
                    complexity_info = self._profile_training_flops(
                        model, reference_batch, flops_batch_size
                    )
                except Exception as error:
                    logger.exception("FLOPs profiling failed")
                    complexity_info = {
                        "error": str(error),
                        "note": "Parameter, memory, and time statistics are still valid.",
                    }

            del optimizer, criterion, model
            gc.collect()
            torch.cuda.empty_cache()

        summary_keys = (
            "mean_data_time_ms",
            "mean_compute_time_ms",
            "mean_step_time_ms",
            "samples_per_second",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "estimated_epoch_minutes",
            "estimated_training_hours",
        )
        summary_stats = {
            key: self._summarize_profile_values(runs, key) for key in summary_keys
        }

        gpu_properties = torch.cuda.get_device_properties(self.device)
        payload = {
            "scope": {
                "training_path": "main forward + diversity forward + loss + backward + optimizer",
                "includes_data_loading": True,
                "excludes": ["validation", "checkpoint I/O", "model initialization"],
                "repeats": repeats,
                "warmup_steps_per_repeat": warmup_steps,
                "measured_steps_per_repeat": measure_steps,
                "full_epoch_per_repeat": requested_measure_steps == 0,
                "batches_per_epoch": batches_per_epoch,
                "configured_epochs": self.epochs,
                "profile_cur_stage": cur_stage,
                "configured_batch_size": self.kwargs.get("batch_size"),
                "max_seq_len": self.max_seq_len,
                "window_size": self.window_size,
            },
            "environment": {
                "gpu": torch.cuda.get_device_name(self.device),
                "gpu_total_memory_gib": gpu_properties.total_memory / (1024 ** 3),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "parameters": parameter_info,
            "complexity": complexity_info,
            "runs": runs,
            "summary": summary_stats,
        }

        json_path = output_dir / "motion_flow_fft_training_profile.json"
        csv_path = output_dir / "motion_flow_fft_training_profile.csv"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

        csv_fields = [
            "row_type",
            "repeat",
            "seed",
            "measured_steps",
            "mean_data_time_ms",
            "mean_compute_time_ms",
            "mean_step_time_ms",
            "samples_per_second",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "estimated_epoch_minutes",
            "estimated_training_hours",
            "total_params",
            "trainable_params",
            "estimated_training_step_gflops_per_sample",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            writer.writeheader()
            for run in runs:
                row = {field: run.get(field, "") for field in csv_fields}
                row.update({
                    "row_type": "run",
                    "total_params": parameter_info["total_params"],
                    "trainable_params": parameter_info["trainable_params"],
                    "estimated_training_step_gflops_per_sample": complexity_info.get(
                        "estimated_training_step_gflops_per_sample", ""
                    ),
                })
                writer.writerow(row)

            mean_row = {field: "" for field in csv_fields}
            mean_row.update({
                "row_type": "mean",
                "repeat": repeats,
                "measured_steps": measure_steps,
                "total_params": parameter_info["total_params"],
                "trainable_params": parameter_info["trainable_params"],
                "estimated_training_step_gflops_per_sample": complexity_info.get(
                    "estimated_training_step_gflops_per_sample", ""
                ),
            })
            for key in summary_keys:
                mean_row[key] = summary_stats[key]["mean"]
            writer.writerow(mean_row)

        logger.info("Three-run mean training resource profile:")
        logger.info(
            "step=%.3f ms, compute=%.3f ms, throughput=%.3f samples/s",
            summary_stats["mean_step_time_ms"]["mean"],
            summary_stats["mean_compute_time_ms"]["mean"],
            summary_stats["samples_per_second"]["mean"],
        )
        logger.info(
            "peak allocated=%.1f MiB, peak reserved=%.1f MiB",
            summary_stats["peak_allocated_mib"]["mean"],
            summary_stats["peak_reserved_mib"]["mean"],
        )
        logger.info(
            "estimated epoch=%.3f min, estimated %d-epoch training=%.3f h",
            summary_stats["estimated_epoch_minutes"]["mean"],
            self.epochs,
            summary_stats["estimated_training_hours"]["mean"],
        )
        logger.info("Saved training profile JSON to %s", json_path)
        logger.info("Saved training profile CSV to %s", csv_path)

    def profile_model(self):
        # --- 新增：设置专用日志文件 ---
        id = self.kwargs.get("resume_runid", "default_runid")
        log_dir = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir"))) / "profile_logs" / id
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / "model_profile.log"
        print(f"Profiling results will be saved to: {log_file}")

        # 创建独立 logger
        profiler_logger = logging.getLogger("profiler")
        profiler_logger.setLevel(logging.INFO)

        # 避免重复添加 handler
        if not profiler_logger.handlers:
            fh = logging.FileHandler(log_file, mode='w')
            formatter = logging.Formatter('%(message)s')  # 只记录消息，不加时间/级别
            fh.setFormatter(formatter)
            profiler_logger.addHandler(fh)
            # 可选：也输出到控制台
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            profiler_logger.addHandler(ch)

        # 替换所有 print 为 profiler_logger.info
        out = profiler_logger.info  # 简化调用
        # ----------------------------------

        model = instantiate(self.model_cfg, _recursive_=False)
        checkpoint_path = self.get_ckpt_path(model, runid="resume_runid", epoch=20)
        from_pretrained_checkpoint(checkpoint_path, model, self.device)
        model.eval()
        model.to(self.device)

        history_kwargs = {"history_mode": self.inference_history, "history_weight": self.history_weight}

        # 获取一个测试 batch 作为示例输入
        test_loader = self.data_module.get_dataloader(stage="test")
        batch = next(iter(test_loader))

        # 解包输入（根据你的 test() 中的输入格式）
        speaker_audio_clips, _, _, speaker_3dmm_clips, _, _, _, speaker_clip_lengths, _ = batch

        print("speaker_audio_clips length:", speaker_audio_clips[0].shape)

        # 取第一个样本并处理成模型所需格式（注意：model 可能接受 list of tensors）
        max_seq_len = self.max_seq_len
        speaker_clip_length = speaker_clip_lengths[0]

        # 模拟 online 分段输入（与 test() 逻辑一致）
        speaker_audio_list = []
        speaker_3dmm_list = []
        motion_length_list = []
        for k in range(math.ceil(speaker_clip_length / max_seq_len)):
            start_idx = k * max_seq_len
            end_idx = min((k + 1) * max_seq_len, speaker_clip_length)
            L = end_idx - start_idx
            motion_length_list.append(L)
            speaker_audio_list.append(speaker_audio_clips[0][start_idx:end_idx].unsqueeze(0))  # [1, L, D]
            speaker_3dmm_list.append(speaker_3dmm_clips[0][start_idx:end_idx].unsqueeze(0))  # [1, L, ...]

        # total_length = s

        if self.model_cfg.task == 'online':
            speaker_audio_list[-1] = self.pad_to(speaker_audio_list[-1].squeeze(0), max_seq_len).unsqueeze(0)
            speaker_3dmm_list[-1] = self.pad_to(speaker_3dmm_list[-1].squeeze(0), max_seq_len).unsqueeze(0)

        motion_lengths = torch.tensor(motion_length_list).to(self.device)
        speaker_audio_input = [x.squeeze(0).to(self.device) for x in speaker_audio_list]
        speaker_3dmm_input = [x.squeeze(0).to(self.device) for x in speaker_3dmm_list]

        # === 1. 参数量 ===
        out("=" * 50)
        out("Model Summary (Parameters):")
        try:
            summary_result = summary(
                model,
                input_data=(speaker_3dmm_input, speaker_audio_input),
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=2,
                device=self.device,
                verbose=0,
                **history_kwargs,
            )
            # # 将 summary 转为字符串并逐行输出
            # summary_str = str(summary_result)
            # for line in summary_str.split('\n'):
            #     out(line)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            out(f"Total params: {total_params:,}")
            out(f"Trainable params: {trainable_params:,}")
        except Exception as e:
            out(f"torchinfo failed: {e}")
            total_params = sum(p.numel() for p in model.parameters())
            out(f"Total params (fallback): {total_params:,}")

        # === 2. FLOPs ===
        out("\n" + "=" * 50)
        out("FLOPs Analysis:")

        try:
            # 保存原始 forward
            original_forward = model.forward

            # 定义一个适配器：接收 ()，内部调用实际 forward
            def forward_for_flops(*args, **kwargs):
                out("Start FLOPs forward")

                return original_forward(
                    speaker_3dmm_input,
                    speaker_audio_input,
                    motion_lengths=motion_lengths,
                    **history_kwargs,
                )

            # 临时替换 forward（FlopCountAnalysis 只调用 model()）
            model.forward = forward_for_flops

            # 现在传入 model 本身（是 nn.Module）
            flops = FlopCountAnalysis(model, ())  # 注意：输入是空 tuple，因为 forward_for_flops 不需要参数
            flop_table = flop_count_table(flops, max_depth=4)
            for line in flop_table.split('\n'):
                out(line)
            total_flops = flops.total()
            out(f"Total FLOPs: {total_flops:,} ({total_flops / 1e9:.2f} GFLOPs)")

            # 恢复原始 forward
            model.forward = original_forward

        except Exception as e:
            out(f"FLOPs analysis failed: {e}")

        # === 3. FPS ===
        out("\n" + "=" * 50)
        out("FPS Measurement:")
        warmup = 10
        trials = 100

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(speaker_3dmm_input, speaker_audio_input, motion_lengths=motion_lengths, **history_kwargs)

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(trials):
                _ = model(speaker_3dmm_input, speaker_audio_input, motion_lengths=motion_lengths, **history_kwargs)
        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_inference = total_time / trials
        total_frames = sum(motion_length_list)
        fps = total_frames / avg_time_per_inference

        out(f"Total frames per sequence: {total_frames}")
        out(f"Avg inference time per sequence: {avg_time_per_inference:.4f} s")
        out(f"FPS (frames per second): {fps:.2f}")

        out(f"\nProfile log saved to: {log_file}")


    def save_to_avi(self, speaker_video_clip, output_path, fps=30):
        """
        speaker_video_clip: Tensor of shape (T, 3, H, W)
        output_path: path to save the avi file
        fps: frames per second
        """
        # 处理 PyTorch 张量
        if isinstance(speaker_video_clip, torch.Tensor):
            speaker_video_clip = speaker_video_clip.detach().cpu().numpy()

        # 转换形状: (T, C, H, W) -> (T, H, W, C)
        video_frames = np.transpose(speaker_video_clip, (0, 2, 3, 1))  # (T, 224, 224, 3)

        # 关键：反归一化！假设原始 Transform 用了 Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        # 即：normalized = (original / 255.0 - 0.5) / 0.5 = original/127.5 - 1 → range [-1, 1]
        # 反推：original = (normalized + 1) * 127.5
        if video_frames.dtype in [np.float32, np.float64]:
            # 检查是否为 [-1, 1] 范围（典型归一化结果）
            if video_frames.min() < 0:  # 很可能就是 [-1,1]
                video_frames = ((video_frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            elif video_frames.max() <= 1.0:  # [0,1]
                video_frames = (video_frames * 255.0).clip(0, 255).astype(np.uint8)
            else:
                video_frames = np.clip(video_frames, 0, 255).astype(np.uint8)
        else:
            video_frames = np.clip(video_frames, 0, 255).astype(np.uint8)

        T, H, W, C = video_frames.shape
        assert C == 3, "Expected 3 channels"

        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        for frame in video_frames:
            # 必须：RGB -> BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
