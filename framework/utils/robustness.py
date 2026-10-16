"""Input perturbations used by the Motion Flow FFT robustness evaluation."""

from typing import Iterable, List, Sequence, Tuple

import torch


SUPPORTED_ROBUSTNESS_MODES = {
    "clean",
    "audio_zero",
    "visual_zero",
    "expression_zero",
    "pose_zero",
    "all_zero",
    "hard_zero",
    "speaker_condition_zero",
    "speaker_feature_zero",
    "history_zero",
    "flow_noise_zero",
    "flow_latent_zero",
    "periodic_zero",
    "audio_noise",
    "audio_mismatch",
}


def validate_robustness_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in SUPPORTED_ROBUSTNESS_MODES:
        supported = ", ".join(sorted(SUPPORTED_ROBUSTNESS_MODES))
        raise ValueError(f"Unsupported robustness mode '{mode}'. Choose from: {supported}")
    return mode


def robustness_condition_name(mode: str, audio_snr_db: float) -> str:
    mode = validate_robustness_mode(mode)
    if mode != "audio_noise":
        return mode
    value = float(audio_snr_db)
    value_text = str(int(value)) if value.is_integer() else str(value).replace(".", "p")
    return f"audio_noise_{value_text}db"


def _valid_length(length, maximum: int) -> int:
    if isinstance(length, torch.Tensor):
        length = length.item()
    return max(0, min(int(length), maximum))


def add_feature_noise_at_snr(
        audio: torch.Tensor,
        valid_length: int,
        snr_db: float,
        eps: float = 1e-12,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise at a requested feature-space SNR."""
    output = audio.clone()
    valid_length = _valid_length(valid_length, output.shape[0])
    if valid_length == 0:
        return output

    signal = output[:valid_length]
    signal_float = signal.float()
    signal_power = signal_float.square().mean()
    if signal_power <= eps:
        return output

    noise = torch.randn_like(signal_float)
    noise_power = noise.square().mean().clamp_min(eps)
    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = noise * torch.sqrt(target_noise_power / noise_power)
    output[:valid_length] = (signal_float + noise).to(dtype=output.dtype)
    return output


def compute_audio_noise_audit(
        clean_audio_clips: Sequence[torch.Tensor],
        noisy_audio_clips: Sequence[torch.Tensor],
        speaker_clip_lengths: Sequence,
        requested_snr_db: float,
        eps: float = 1e-12,
) -> dict:
    """Measure the SNR actually applied to the Wav2Vec feature tensors."""
    measured = []
    for clean, noisy, length in zip(
            clean_audio_clips, noisy_audio_clips, speaker_clip_lengths
    ):
        valid_length = _valid_length(length, min(clean.shape[0], noisy.shape[0]))
        if valid_length == 0:
            continue
        signal = clean[:valid_length].float()
        noise = noisy[:valid_length].float() - signal
        signal_power = signal.square().mean()
        noise_power = noise.square().mean()
        if signal_power <= eps or noise_power <= eps:
            continue
        measured.append(float(10.0 * torch.log10(signal_power / noise_power).item()))

    return {
        "domain": "wav2vec_feature",
        "distribution": "independent_gaussian",
        "requested_snr_db": float(requested_snr_db),
        "measured_snr_db_mean": (
            sum(measured) / len(measured) if measured else None
        ),
        "measured_snr_db_min": min(measured) if measured else None,
        "measured_snr_db_max": max(measured) if measured else None,
        "measured_clips": len(measured),
    }


def apply_input_perturbation(
        speaker_audio_clips: Sequence[torch.Tensor],
        speaker_3dmm_clips: Sequence[torch.Tensor],
        speaker_clip_lengths: Sequence,
        mode: str,
        audio_snr_db: float = 20.0,
        expression_dims: int = 52,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Apply one controlled perturbation while keeping all other inputs fixed."""
    mode = validate_robustness_mode(mode)
    audio_outputs = list(speaker_audio_clips)
    motion_outputs = list(speaker_3dmm_clips)

    if mode in {
        "clean",
        "audio_mismatch",
        "speaker_feature_zero",
        "history_zero",
        "flow_noise_zero",
        "flow_latent_zero",
        "periodic_zero",
    }:
        return audio_outputs, motion_outputs
    if mode in {"all_zero", "hard_zero", "speaker_condition_zero"}:
        audio_outputs = [torch.zeros_like(audio) for audio in speaker_audio_clips]
        motion_outputs = [torch.zeros_like(motion) for motion in speaker_3dmm_clips]
    elif mode == "audio_zero":
        audio_outputs = [torch.zeros_like(audio) for audio in speaker_audio_clips]
    elif mode == "visual_zero":
        motion_outputs = [torch.zeros_like(motion) for motion in speaker_3dmm_clips]
    elif mode == "expression_zero":
        motion_outputs = []
        for motion in speaker_3dmm_clips:
            changed = motion.clone()
            changed[..., :min(expression_dims, changed.shape[-1])] = 0
            motion_outputs.append(changed)
    elif mode == "pose_zero":
        motion_outputs = []
        for motion in speaker_3dmm_clips:
            changed = motion.clone()
            changed[..., min(expression_dims, changed.shape[-1]):] = 0
            motion_outputs.append(changed)
    elif mode == "audio_noise":
        audio_outputs = [
            add_feature_noise_at_snr(audio, length, audio_snr_db)
            for audio, length in zip(speaker_audio_clips, speaker_clip_lengths)
        ]

    return audio_outputs, motion_outputs


def compute_input_audit(
        speaker_audio_clips: Sequence[torch.Tensor],
        speaker_3dmm_clips: Sequence[torch.Tensor],
        expression_dims: int = 52,
) -> dict:
    """Summarize post-perturbation tensors for runtime verification."""

    def summarize(items: Sequence[torch.Tensor]) -> dict:
        valid_items = [item for item in items if item.numel() > 0]
        return {
            "max_abs": max((float(item.abs().max().item()) for item in valid_items), default=0.0),
            "nonzero_count": int(sum(torch.count_nonzero(item).item() for item in valid_items)),
            "numel": int(sum(item.numel() for item in valid_items)),
        }

    expression = [motion[..., :min(expression_dims, motion.shape[-1])] for motion in speaker_3dmm_clips]
    pose = [motion[..., min(expression_dims, motion.shape[-1]):] for motion in speaker_3dmm_clips]
    return {
        "audio": summarize(speaker_audio_clips),
        "3dmm": summarize(speaker_3dmm_clips),
        "expression": summarize(expression),
        "pose": summarize(pose),
    }


def clone_tensor_list(items: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    return [item.detach().clone() for item in items]


def _fit_sequence_length(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    if sequence.shape[0] >= target_length:
        return sequence[:target_length].clone()
    pad_shape = (target_length - sequence.shape[0], *sequence.shape[1:])
    return torch.cat((sequence, sequence.new_zeros(pad_shape)), dim=0)


def mismatch_audio_from_previous_batch(
        current_audio_clips: Sequence[torch.Tensor],
        previous_audio_clips: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Pair each current sample with an audio tensor from the previous batch."""
    if not previous_audio_clips:
        raise ValueError("previous_audio_clips must not be empty")
    return [
        _fit_sequence_length(
            previous_audio_clips[index % len(previous_audio_clips)],
            current.shape[0],
        )
        for index, current in enumerate(current_audio_clips)
    ]


def compute_expression_activity(
        speaker_3dmm: torch.Tensor,
        valid_length: int,
        expression_dims: int = 52,
) -> float:
    """Mean temporal variance of the valid expression coefficients."""
    valid_length = _valid_length(valid_length, speaker_3dmm.shape[0])
    if valid_length < 2:
        return 0.0
    expression = speaker_3dmm[:valid_length, :expression_dims].float()
    return float(expression.var(dim=0, unbiased=False).mean().item())


def save_robustness_debug_visualization(
        debug_tensors: dict,
        output_path,
        expression_dims: int = 52,
        max_frames: int = 128,
) -> dict:
    """Save per-stage temporal RMS curves and return auditable tensor statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prepared = {}
    speaker_3dmm = debug_tensors.get("speaker_3dmm")
    if isinstance(speaker_3dmm, torch.Tensor):
        split_at = min(expression_dims, speaker_3dmm.shape[-1])
        prepared["speaker_expression"] = speaker_3dmm[..., :split_at]
        prepared["speaker_pose"] = speaker_3dmm[..., split_at:]

    display_order = (
        ("speaker_audio", "Speaker Wav2Vec input"),
        ("speaker_expression", "Speaker 3DMM expression"),
        ("speaker_pose", "Speaker 3DMM pose"),
        ("speaker_feature", "Speaker encoded feature"),
        ("listener_history_3dmm", "Listener history 3DMM"),
        ("encoded_feature", "Fused condition feature"),
        ("flow_latent", "Flow output latent"),
        ("periodic_feature", "Periodic feature"),
        ("prediction_3dmm", "Predicted listener 3DMM"),
    )
    for name, _ in display_order:
        if name not in prepared and isinstance(debug_tensors.get(name), torch.Tensor):
            prepared[name] = debug_tensors[name]

    stats = {}
    figure, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    for axis, (name, title) in zip(axes.flat, display_order):
        tensor = prepared.get(name)
        if tensor is None or tensor.numel() == 0:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            axis.set_title(title)
            axis.set_axis_off()
            stats[name] = {"available": False}
            continue

        value = tensor.detach().float().cpu()
        while value.ndim > 2:
            value = value[0]
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        value = value[:max_frames]
        rms_curve = value.square().mean(dim=-1).sqrt()
        nonzero_count = int(torch.count_nonzero(value).item())
        stats[name] = {
            "available": True,
            "shape": list(tensor.shape),
            "max_abs": float(value.abs().max().item()),
            "mean_abs": float(value.abs().mean().item()),
            "rms": float(value.square().mean().sqrt().item()),
            "nonzero_ratio": nonzero_count / value.numel(),
        }

        axis.plot(rms_curve.numpy(), linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Frame")
        axis.set_ylabel("RMS")
        axis.grid(alpha=0.25)
        axis.text(
            0.02,
            0.95,
            f"max={stats[name]['max_abs']:.4g}\nrms={stats[name]['rms']:.4g}\n"
            f"nonzero={100.0 * stats[name]['nonzero_ratio']:.1f}%",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )

    figure.suptitle("Motion Flow FFT inference signal audit", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return stats
