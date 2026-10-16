"""Evaluate the 3DMM-to-emotion converter with ground-truth 3DMM inputs.

This is the oracle experiment used to quantify the domain gap introduced by
the frozen Emotion3DMMTransformer. It intentionally reuses the repository's
ReactionDataset and compute_metrics implementation.
"""

import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.react_2025 import custom_collate
from framework.utils.compute_metrics import compute_metrics


logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_converter(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    converter = instantiate(cfg.converter)
    checkpoint_path = Path(to_absolute_path(cfg.checkpoint))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Converter checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    converter.load_state_dict(state_dict, strict=True)
    converter.to(device)
    converter.eval()
    converter.reset_window_size(int(cfg.window_size))
    converter.set_inference()

    for parameter in converter.parameters():
        parameter.requires_grad_(False)

    logger.info("Loaded frozen converter: %s", checkpoint_path)
    return converter


def validate_batch_shapes(
    listener_3dmm: torch.Tensor,
    converted_emotion: torch.Tensor,
    target_emotion: torch.Tensor,
) -> None:
    if listener_3dmm.ndim != 3 or listener_3dmm.shape[-1] != 58:
        raise ValueError(
            f"Expected listener 3DMM [B,T,58], got {tuple(listener_3dmm.shape)}"
        )
    if converted_emotion.ndim != 3 or converted_emotion.shape[-1] != 25:
        raise ValueError(
            "Expected converted emotion [B,T,25], got "
            f"{tuple(converted_emotion.shape)}"
        )
    if converted_emotion.shape != target_emotion.shape:
        raise ValueError(
            "Converted and target emotion shapes differ: "
            f"{tuple(converted_emotion.shape)} vs {tuple(target_emotion.shape)}"
        )
    if not torch.isfinite(converted_emotion).all():
        raise ValueError("Converted emotion contains NaN or Inf values")


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="domain_gap_upper_bound",
)
def main(cfg: DictConfig) -> None:
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))
    # The metric workers receive many independent CPU tensors. The default
    # file_descriptor strategy can exceed Linux's open-file limit on the full
    # validation set, while file_system preserves the same metric semantics.
    torch.multiprocessing.set_sharing_strategy("file_system")

    requested_device = str(cfg.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    dataset = instantiate(cfg.dataset)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=custom_collate,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    converter = load_converter(cfg, device)

    speaker_emotions = []
    converted_predictions = []
    listener_targets = []
    listener_3dmm_gt = []
    max_samples = int(cfg.max_samples)

    for batch in tqdm(loader, desc="GT 3DMM -> emotion"):
        speaker_emotion_batch = batch[2]
        listener_emotion_batch = batch[5]
        listener_3dmm_batch = batch[6]

        gt_3dmm = torch.stack(listener_3dmm_batch).float().to(device)
        gt_emotion = torch.stack(listener_emotion_batch).float()

        with torch.inference_mode():
            converted_emotion = converter.to_emotion(gt_3dmm)[0].cpu()

        if bool(cfg.data_clamp):
            # Tensors created in inference_mode cannot be updated in place
            # after leaving that context. Build a regular output tensor.
            converted_emotion = torch.cat(
                (
                    torch.round(converted_emotion[..., :15]),
                    converted_emotion[..., 15:],
                ),
                dim=-1,
            )

        validate_batch_shapes(gt_3dmm, converted_emotion, gt_emotion)

        for sample_index in range(converted_emotion.shape[0]):
            prediction = converted_emotion[sample_index]
            target = gt_emotion[sample_index]
            speaker = speaker_emotion_batch[sample_index].float()

            # Prediction: [10,T,25]. Validation has one GT: [1,T,25].
            converted_predictions.append(
                prediction.unsqueeze(0)
                .repeat(int(cfg.num_preds), 1, 1)
                .contiguous()
            )
            listener_targets.append(target.unsqueeze(0).contiguous())
            speaker_emotions.append(speaker.contiguous())
            listener_3dmm_gt.append(gt_3dmm[sample_index].cpu())

            if max_samples > 0 and len(converted_predictions) >= max_samples:
                break

        if max_samples > 0 and len(converted_predictions) >= max_samples:
            break

    if not converted_predictions:
        raise RuntimeError("The validation dataset produced no samples")

    output_dir = Path(to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "domain_gap_outputs.pt"
    torch.save(
        {
            "speaker_emotion": speaker_emotions,
            "prediction_emotion": converted_predictions,
            "target_emotion": listener_targets,
            "listener_3dmm_gt": listener_3dmm_gt,
            "metadata": {
                "split": str(cfg.dataset.split),
                "clip_length": int(cfg.clip_length),
                "num_preds": int(cfg.num_preds),
                "window_size": int(cfg.window_size),
                "normalize_3dmm": str(cfg.dataset.normalize_3dmm),
                "data_clamp": bool(cfg.data_clamp),
                "checkpoint": str(Path(to_absolute_path(cfg.checkpoint))),
            },
        },
        tensor_path,
    )

    results = compute_metrics(
        speaker_emotions,
        converted_predictions,
        listener_targets,
        threads=int(cfg.metric_threads),
    )
    serializable_results = {key: float(value) for key, value in results.items()}
    serializable_results["num_samples"] = len(converted_predictions)

    metrics_path = output_dir / "domain_gap_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_results, file, indent=2, ensure_ascii=False)

    logger.info(
        "Oracle metrics: FRCorr/FRC=%.6f, FRDist/FRD=%.6f, TLCC=%.6f",
        serializable_results["FRC"],
        serializable_results["FRD"],
        serializable_results["TLCC"],
    )
    logger.info("Saved tensors to %s", tensor_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info(
        "FRVar and smse are diagnostic only; identical repeated oracle "
        "predictions do not define a diversity upper bound."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
