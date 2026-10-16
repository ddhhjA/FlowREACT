"""Benchmark raw audio to frame-aligned Wav2Vec features.

Run this script from the repository root. It measures serial batch-size-one
latency, so every timed iteration is synchronized on CUDA.
"""

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.modules.audio_processor import AudioProcessor


SUPPORTED_MODES = (
    "legacy",
    "wav2vec_only",
    "audio_to_vec",
    "end_to_end",
    "stream_window",
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fixed_length_waveform(waveform, frames, sample_rate, fps):
    """Crop or zero-pad a waveform to the audio duration of `frames`."""
    num_samples = max(1, int(frames * sample_rate / fps))
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.shape[0] >= num_samples:
        return np.ascontiguousarray(waveform[:num_samples])
    return np.pad(waveform, (0, num_samples - waveform.shape[0])).astype(np.float32)


def discover_audio_files(data_dir, split, audio_dir=None):
    if audio_dir is not None:
        candidates = [Path(audio_dir)]
    else:
        root = Path(data_dir)
        candidates = [root / split / "audio", root / "audio", root]

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        files = sorted(candidate.rglob("*.wav"))
        if files:
            return candidate, files
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No WAV files found. Searched: {searched}")


def select_and_load_waveforms(processor, files, frames, max_files, seed):
    files = list(files)
    random.Random(seed).shuffle(files)
    if max_files > 0:
        files = files[:max_files]

    valid_files = []
    waveforms = []
    for wav_file in files:
        try:
            waveform, _ = processor.load_waveform(str(wav_file))
        except Exception as error:  # Keep a long benchmark usable if one file is corrupt.
            print(f"[skip] {wav_file}: {error}")
            continue
        valid_files.append(wav_file)
        waveforms.append(
            fixed_length_waveform(waveform, frames, processor.sample_rate, processor.fps)
        )

    if not waveforms:
        raise RuntimeError("No readable WAV files are available for benchmarking")
    return valid_files, waveforms


def prepare_tensor(processor, waveform, frames):
    audio_feature, encoded_seq_len, _ = processor.prepare_waveform(
        waveform,
        processor.sample_rate,
        seq_len=frames,
        warn_length_mismatch=False,
    )
    if encoded_seq_len != frames:
        raise RuntimeError(f"Expected {frames} output frames, got {encoded_seq_len}")
    return audio_feature


def validate_output(output, frames, hidden_size):
    expected_shape = (frames, hidden_size)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Expected output shape {expected_shape}, got {tuple(output.shape)}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Wav2Vec output contains NaN or Inf")


def check_last_hidden_state_equivalence(processor, audio_feature, frames, precision):
    """Check that disabling unused hidden-state collection preserves the output."""
    device_input = audio_feature.to(processor.device)
    with torch.inference_mode():
        reference = processor.audio_encoder(
            device_input,
            seq_len=frames,
            output_attentions=False,
            output_hidden_states=True,
            verbose=False,
        ).last_hidden_state.squeeze(0)
    candidate = processor.encode_waveform(
        device_input,
        seq_len=frames,
        return_cpu=False,
        precision=precision,
    )
    synchronize(processor.device)
    max_abs_diff = (reference.float() - candidate.float()).abs().max().item()
    if precision == "fp32" and not torch.allclose(reference, candidate, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            f"Optimized FP32 output differs from the reference; max abs diff={max_abs_diff:.6g}"
        )
    return max_abs_diff


def t_critical_95(sample_count):
    # Two-sided Student-t critical values for 95% CIs. The benchmark defaults to n=5.
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    return values.get(sample_count, 1.96)


def aggregate(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "sd": 0.0, "ci95": [mean, mean]}
    sd = statistics.stdev(values)
    margin = t_critical_95(len(values)) * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "ci95": [mean - margin, mean + margin]}


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def build_mode_runner(
        mode,
        processor,
        files,
        full_waveforms,
        full_gpu_tensor,
        clip_frames,
        window_frames,
        precision,
):
    if mode == "legacy":
        frames = clip_frames

        def run(index):
            waveform, _ = processor.load_waveform(str(files[index % len(files)]))
            waveform = fixed_length_waveform(
                waveform,
                frames,
                processor.sample_rate,
                processor.fps,
            )
            audio_feature = prepare_tensor(processor, waveform, frames).to(processor.device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if precision == "fp16"
                else nullcontext()
            )
            with torch.inference_mode():
                with amp_context:
                    output = processor.audio_encoder(
                        audio_feature,
                        seq_len=frames,
                        output_attentions=False,
                        output_hidden_states=True,
                        verbose=False,
                    ).last_hidden_state.squeeze(0)
            return output.cpu()

    elif mode == "wav2vec_only":
        frames = clip_frames

        def run(index):
            del index
            return processor.encode_waveform(
                full_gpu_tensor,
                seq_len=frames,
                return_cpu=False,
                precision=precision,
            )

    elif mode == "audio_to_vec":
        frames = clip_frames

        def run(index):
            waveform = full_waveforms[index % len(full_waveforms)]
            audio_feature = prepare_tensor(processor, waveform, frames)
            return processor.encode_waveform(
                audio_feature,
                seq_len=frames,
                return_cpu=False,
                precision=precision,
            )

    elif mode == "end_to_end":
        frames = clip_frames

        def run(index):
            waveform, _ = processor.load_waveform(str(files[index % len(files)]))
            waveform = fixed_length_waveform(
                waveform,
                frames,
                processor.sample_rate,
                processor.fps,
            )
            audio_feature = prepare_tensor(processor, waveform, frames)
            return processor.encode_waveform(
                audio_feature,
                seq_len=frames,
                return_cpu=True,
                precision=precision,
            )

    elif mode == "stream_window":
        frames = window_frames
        window_waveforms = [
            fixed_length_waveform(
                waveform,
                frames,
                processor.sample_rate,
                processor.fps,
            )
            for waveform in full_waveforms
        ]

        def run(index):
            audio_feature = prepare_tensor(
                processor,
                window_waveforms[index % len(window_waveforms)],
                frames,
            )
            return processor.encode_waveform(
                audio_feature,
                seq_len=frames,
                return_cpu=False,
                precision=precision,
            )

    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    return frames, run


def benchmark_mode(
        mode,
        processor,
        files,
        full_waveforms,
        full_gpu_tensor,
        clip_frames,
        window_frames,
        precision,
        warmup,
        repeat,
        runs,
):
    frames, run_once = build_mode_runner(
        mode,
        processor,
        files,
        full_waveforms,
        full_gpu_tensor,
        clip_frames,
        window_frames,
        precision,
    )
    hidden_size = processor.audio_encoder.config.hidden_size

    for index in range(warmup):
        output = run_once(index)
    synchronize(processor.device)
    validate_output(output, frames, hidden_size)

    run_fps = []
    run_rtf = []
    run_latency_ms = []
    item_latency_ms = []
    peak_memory_bytes = []

    for run_index in range(runs):
        if processor.device.type == "cuda":
            synchronize(processor.device)
            torch.cuda.reset_peak_memory_stats(processor.device)

        timings = []
        for item_index in range(repeat):
            synchronize(processor.device)
            start = time.perf_counter()
            output = run_once(run_index * repeat + item_index)
            synchronize(processor.device)
            timings.append(time.perf_counter() - start)

        validate_output(output, frames, hidden_size)
        elapsed = sum(timings)
        total_frames = frames * repeat
        total_audio_seconds = total_frames / processor.fps
        run_fps.append(total_frames / elapsed)
        run_rtf.append(elapsed / total_audio_seconds)
        run_latency_ms.append(elapsed * 1000.0 / repeat)
        item_latency_ms.extend(value * 1000.0 for value in timings)
        if processor.device.type == "cuda":
            peak_memory_bytes.append(torch.cuda.max_memory_allocated(processor.device))

    fps_stats = aggregate(run_fps)
    latency_stats = aggregate(run_latency_ms)
    frame_latency_stats = aggregate([value / frames for value in run_latency_ms])
    rtf_stats = aggregate(run_rtf)
    result = {
        "mode": mode,
        "frames_per_item": frames,
        "audio_seconds_per_item": frames / processor.fps,
        "precision": precision,
        "warmup": warmup,
        "repeat_per_run": repeat,
        "runs": runs,
        "fps": fps_stats,
        "latency_ms_per_item": latency_stats,
        "latency_ms_per_frame": frame_latency_stats,
        "latency_ms_p50": percentile(item_latency_ms, 50),
        "latency_ms_p95": percentile(item_latency_ms, 95),
        "rtf": rtf_stats,
        "audio_seconds_per_wall_second": 1.0 / rtf_stats["mean"],
        "peak_gpu_memory_mb": (
            max(peak_memory_bytes) / (1024 ** 2) if peak_memory_bytes else 0.0
        ),
        "per_run_fps": run_fps,
        "per_run_latency_ms": run_latency_ms,
        "per_run_rtf": run_rtf,
    }
    return result


def system_metadata(device, args, audio_root, file_count, model_load_seconds):
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "audio_root": str(audio_root),
        "readable_audio_files": file_count,
        "model_path": str(Path(args.wav2vec_model_path).resolve()),
        "model_load_seconds": model_load_seconds,
        "sample_rate": args.sample_rate,
        "output_fps": args.fps,
        "clip_frames": args.clip_frames,
        "window_frames": args.window_frames,
        "seed": args.seed,
        "cpu_threads": torch.get_num_threads(),
        "timing_scope": "serial batch-size-one latency with per-item CUDA synchronization",
        "file_io_note": "end_to_end includes WAV decode/resample; OS file cache may be warm",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update({
            "gpu_name": properties.name,
            "gpu_total_memory_gb": properties.total_memory / (1024 ** 3),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
        })
    return metadata


def save_results(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    csv_path = output_path.with_suffix(".csv")
    fieldnames = [
        "mode",
        "frames_per_item",
        "precision",
        "fps_mean",
        "fps_sd",
        "fps_ci95_low",
        "fps_ci95_high",
        "latency_ms_mean",
        "latency_ms_per_frame",
        "latency_ms_p50",
        "latency_ms_p95",
        "rtf_mean",
        "audio_seconds_per_wall_second",
        "peak_gpu_memory_mb",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in payload["results"]:
            writer.writerow({
                "mode": result["mode"],
                "frames_per_item": result["frames_per_item"],
                "precision": result["precision"],
                "fps_mean": result["fps"]["mean"],
                "fps_sd": result["fps"]["sd"],
                "fps_ci95_low": result["fps"]["ci95"][0],
                "fps_ci95_high": result["fps"]["ci95"][1],
                "latency_ms_mean": result["latency_ms_per_item"]["mean"],
                "latency_ms_per_frame": result["latency_ms_per_frame"]["mean"],
                "latency_ms_p50": result["latency_ms_p50"],
                "latency_ms_p95": result["latency_ms_p95"],
                "rtf_mean": result["rtf"]["mean"],
                "audio_seconds_per_wall_second": result["audio_seconds_per_wall_second"],
                "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            })
    return output_path, csv_path


def parse_modes(value):
    if value.strip().lower() == "all":
        return list(SUPPORTED_MODES)
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in modes if item not in SUPPORTED_MODES]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}; choose from {SUPPORTED_MODES}")
    if not modes:
        raise ValueError("At least one benchmark mode is required")
    return modes


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if args.clip_frames <= 0 or args.window_frames <= 0:
        raise ValueError("clip_frames and window_frames must be positive")
    if args.repeat <= 0 or args.runs <= 0:
        raise ValueError("repeat and runs must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    audio_root, all_files = discover_audio_files(args.data_dir, args.split, args.audio_dir)
    print(f"Found {len(all_files)} WAV files under {audio_root}")

    model_load_start = time.perf_counter()
    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        fps=args.fps,
        wav2vec_model_path=args.wav2vec_model_path,
        only_last_features=True,
        device=device,
    )
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

    files, full_waveforms = select_and_load_waveforms(
        processor,
        all_files,
        frames=args.clip_frames,
        max_files=args.max_files,
        seed=args.seed,
    )
    full_cpu_tensor = prepare_tensor(processor, full_waveforms[0], args.clip_frames)
    full_gpu_tensor = full_cpu_tensor.to(device=device, non_blocking=True)
    synchronize(device)

    max_abs_diff = check_last_hidden_state_equivalence(
        processor,
        full_cpu_tensor,
        frames=args.clip_frames,
        precision=args.precision,
    )
    print(f"Correctness check max abs diff: {max_abs_diff:.6g}")

    results = []
    for mode in parse_modes(args.modes):
        print(f"\nBenchmarking {mode} ...")
        result = benchmark_mode(
            mode=mode,
            processor=processor,
            files=files,
            full_waveforms=full_waveforms,
            full_gpu_tensor=full_gpu_tensor,
            clip_frames=args.clip_frames,
            window_frames=args.window_frames,
            precision=args.precision,
            warmup=args.warmup,
            repeat=args.repeat,
            runs=args.runs,
        )
        results.append(result)
        print(
            f"{mode}: FPS={result['fps']['mean']:.2f} +/- {result['fps']['sd']:.2f}, "
            f"latency={result['latency_ms_per_item']['mean']:.3f} ms/item, "
            f"{result['latency_ms_per_frame']['mean']:.4f} ms/frame, "
            f"P95={result['latency_ms_p95']:.3f} ms, RTF={result['rtf']['mean']:.4f}"
        )

    payload = {
        "metadata": system_metadata(
            device,
            args,
            audio_root,
            len(files),
            model_load_seconds,
        ),
        "correctness": {
            "output_shape": [args.clip_frames, processor.audio_encoder.config.hidden_size],
            "optimized_vs_hidden_states_reference_max_abs_diff": max_abs_diff,
            "fp32_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        },
        "results": results,
    }
    json_path, csv_path = save_results(args.output, payload)
    print(f"\nSaved JSON: {json_path.resolve()}")
    print(f"Saved CSV:  {csv_path.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark raw WAV to frame-aligned 768D Wav2Vec features"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Optional direct path to the WAV directory; overrides data_dir/split/audio",
    )
    parser.add_argument(
        "--wav2vec_model_path",
        type=str,
        default="./pretrained_models/wav2vec/wav2vec2-base-960h",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip_frames", type=int, default=750)
    parser.add_argument("--window_frames", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--modes",
        type=str,
        default="all",
        help=f"all or a comma-separated subset of: {','.join(SUPPORTED_MODES)}",
    )
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output",
        type=str,
        default="results/audio_vec_profile_fp32.json",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
