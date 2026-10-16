"""Aggregate blink-event duration statistics from 3DMM coefficients."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_vector(path, name):
    value = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if value.ndim != 1:
        value = value.reshape(-1)
    if value.size < 10:
        raise ValueError(f"{name} must contain at least 10 coefficients: {path}")
    return value


def coefficient_transform(normalization, mean_face_path, std_face_path):
    if normalization == "none":
        return None, None
    mean = load_vector(mean_face_path, "mean face")
    if normalization == "zero_center":
        return mean, None
    std = load_vector(std_face_path, "face standard deviation")
    return mean, np.maximum(std, 1e-8)


def as_coefficients(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32).squeeze()
    if value.ndim == 1:
        value = value[None, :]
    elif value.ndim > 2:
        value = value.reshape(-1, value.shape[-1])
    if value.ndim != 2 or value.shape[-1] < 10:
        raise ValueError(f"Expected [T,D] 3DMM coefficients, got {value.shape}")
    return value


def transform_coefficients(value, mean, std):
    value = as_coefficients(value)
    if mean is None:
        return value
    value = value - mean[:value.shape[-1]]
    if std is not None:
        value = value / std[:value.shape[-1]]
    return value


def closure_signal(coefficients, blink_indices):
    left, right = blink_indices
    if coefficients.shape[-1] <= max(left, right):
        raise ValueError(
            f"Blink indices {blink_indices} exceed coefficient dimension {coefficients.shape[-1]}"
        )
    return coefficients[:, [left, right]].mean(axis=-1)


def coefficient_files(root):
    files = sorted(Path(root).rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No coefficient .npy files found under {root}")
    return files


def calibrate_thresholds(
        files,
        blink_indices,
        mean,
        std,
        low_percentile,
        high_percentile,
        stride,
        max_frames,
        seed,
):
    samples = []
    for path in files:
        coefficients = transform_coefficients(np.load(path, mmap_mode="r"), mean, std)
        samples.append(closure_signal(coefficients, blink_indices)[::stride])
    values = np.concatenate(samples).astype(np.float32, copy=False)
    if values.size > max_frames:
        generator = np.random.default_rng(seed)
        indices = generator.choice(values.size, size=max_frames, replace=False)
        values = values[indices]
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high <= low:
        raise RuntimeError(f"Invalid blink calibration range: low={low}, high={high}")
    return {
        "low_value": low,
        "high_value": high,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "sampled_frames": int(values.size),
        "source_files": len(files),
    }


def normalize_closure(signal, calibration):
    denominator = calibration["high_value"] - calibration["low_value"]
    return np.clip((signal - calibration["low_value"]) / denominator, 0.0, 1.0)


def detect_blinks(
        closure,
        start_threshold=0.5,
        end_threshold=0.2,
        merge_gap_frames=2,
        min_event_frames=2,
):
    if not 0.0 <= end_threshold < start_threshold <= 1.0:
        raise ValueError("Require 0 <= end_threshold < start_threshold <= 1")

    raw_events = []
    active = False
    start = None
    left_censored = False
    for frame, value in enumerate(closure):
        if not active and value >= start_threshold:
            active = True
            start = frame
            left_censored = frame == 0
        elif active and value <= end_threshold:
            if not left_censored and frame > start:
                raw_events.append([start, frame])
            active = False
            start = None
            left_censored = False
    # An event still active at the final frame is right-censored and excluded.

    merged = []
    for start, end in raw_events:
        if merged and start - merged[-1][1] <= merge_gap_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    events = []
    for start, end in merged:
        duration_frames = end - start
        if duration_frames < min_event_frames:
            continue
        events.append({
            "start_frame": int(start),
            "end_frame": int(end - 1),
            "duration_frames": int(duration_frames),
            "peak_closure": float(np.max(closure[start:end])),
        })
    return events


def iter_nested_sequences(value, prefix="sequence"):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if array.ndim == 2:
            yield prefix, array
            return
        if array.ndim == 3:
            for index in range(array.shape[0]):
                yield f"{prefix}_{index}", array[index]
            return
        raise ValueError(f"Unsupported prediction tensor shape: {array.shape}")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_nested_sequences(item, f"{prefix}_{index}")
        return
    raise TypeError(f"Unsupported prediction container: {type(value).__name__}")


def bootstrap_cluster_mean(event_sums, event_counts, samples, seed):
    valid = event_counts.sum() > 0
    if not valid or samples <= 0:
        return None
    generator = np.random.default_rng(seed)
    sequence_count = len(event_counts)
    estimates = []
    for _ in range(samples):
        indices = generator.integers(0, sequence_count, size=sequence_count)
        count = event_counts[indices].sum()
        if count > 0:
            estimates.append(event_sums[indices].sum() / count)
    if not estimates:
        return None
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def analyze_sequences(
        sequences,
        method,
        calibration,
        blink_indices,
        fps,
        start_threshold,
        end_threshold,
        merge_gap_frames,
        min_event_frames,
        mean=None,
        std=None,
        bootstrap_samples=2000,
        seed=1234,
        example_limit=3,
):
    rows = []
    event_sums = []
    event_counts = []
    sequence_means = []
    examples = []
    total_frames = 0
    sequence_count = 0

    for sequence_id, value in sequences:
        coefficients = transform_coefficients(value, mean, std)
        closure = normalize_closure(closure_signal(coefficients, blink_indices), calibration)
        events = detect_blinks(
            closure,
            start_threshold=start_threshold,
            end_threshold=end_threshold,
            merge_gap_frames=merge_gap_frames,
            min_event_frames=min_event_frames,
        )
        durations = np.asarray([event["duration_frames"] / fps for event in events], dtype=np.float64)
        event_sums.append(float(durations.sum()))
        event_counts.append(int(durations.size))
        if durations.size:
            sequence_means.append(float(durations.mean()))
        total_frames += len(closure)
        sequence_count += 1

        for event_index, event in enumerate(events):
            rows.append({
                "method": method,
                "sequence_id": sequence_id,
                "event_index": event_index,
                **event,
                "duration_seconds": event["duration_frames"] / fps,
            })
        if len(examples) < example_limit:
            examples.append((sequence_id, closure, events))

    durations = np.asarray([row["duration_seconds"] for row in rows], dtype=np.float64)
    event_sums = np.asarray(event_sums, dtype=np.float64)
    event_counts = np.asarray(event_counts, dtype=np.int64)
    sequence_means = np.asarray(sequence_means, dtype=np.float64)

    def percentile(value):
        return float(value) if durations.size else None

    total_minutes = total_frames / fps / 60.0
    summary = {
        "method": method,
        "sequences": sequence_count,
        "sequences_with_blinks": int(np.count_nonzero(event_counts)),
        "events": int(durations.size),
        "total_minutes": float(total_minutes),
        "mean_duration_seconds": float(durations.mean()) if durations.size else None,
        "std_duration_seconds": float(durations.std(ddof=1)) if durations.size > 1 else 0.0 if durations.size else None,
        "median_duration_seconds": percentile(np.median(durations)) if durations.size else None,
        "q1_duration_seconds": percentile(np.percentile(durations, 25)) if durations.size else None,
        "q3_duration_seconds": percentile(np.percentile(durations, 75)) if durations.size else None,
        "iqr_duration_seconds": percentile(np.percentile(durations, 75) - np.percentile(durations, 25)) if durations.size else None,
        "p95_duration_seconds": percentile(np.percentile(durations, 95)) if durations.size else None,
        "blinks_per_minute": float(durations.size / total_minutes) if total_minutes > 0 else None,
        "mean_sequence_duration_seconds": float(sequence_means.mean()) if sequence_means.size else None,
        "mean_duration_ci95": bootstrap_cluster_mean(
            event_sums, event_counts, bootstrap_samples, seed
        ),
    }
    return summary, rows, durations, examples


def load_results(path, key):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if key not in payload:
        raise KeyError(f"'{key}' not found in {path}; available keys: {sorted(payload.keys())}")
    return payload[key]


def parse_method_inputs(input_specs, legacy_results, legacy_method):
    method_inputs = []
    if legacy_results:
        method_inputs.append((legacy_method, legacy_results))

    for spec in input_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --input value '{spec}'. Expected METHOD=RESULTS_PT"
            )
        method, path = spec.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(
                f"Invalid --input value '{spec}'. Expected METHOD=RESULTS_PT"
            )
        method_inputs.append((method, path))

    if not method_inputs:
        raise ValueError("Provide at least one --input METHOD=RESULTS_PT or --results path")

    methods = [method for method, _ in method_inputs]
    if "Ground Truth" in methods:
        raise ValueError("'Ground Truth' is reserved and cannot be used as a method name")
    duplicates = sorted({method for method in methods if methods.count(method) > 1})
    if duplicates:
        raise ValueError(f"Duplicate method names: {', '.join(duplicates)}")
    return method_inputs


def save_summary_csv(path, summaries):
    fieldnames = [
        "method", "sequences", "sequences_with_blinks", "events", "total_minutes",
        "mean_duration_seconds", "std_duration_seconds", "median_duration_seconds",
        "q1_duration_seconds", "q3_duration_seconds", "iqr_duration_seconds",
        "p95_duration_seconds", "blinks_per_minute", "mean_sequence_duration_seconds",
        "mean_duration_ci95_low", "mean_duration_ci95_high",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            ci = row.pop("mean_duration_ci95")
            row["mean_duration_ci95_low"] = ci[0] if ci else None
            row["mean_duration_ci95_high"] = ci[1] if ci else None
            writer.writerow(row)


def save_events_csv(path, rows):
    fieldnames = [
        "method", "sequence_id", "event_index", "start_frame", "end_frame",
        "duration_frames", "duration_seconds", "peak_closure",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_distribution_plot(path, method_durations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonempty = [value for value in method_durations.values() if value.size]
    if not nonempty:
        return
    upper = max(float(np.percentile(value, 99)) for value in nonempty)
    upper = max(upper, 1.0 / 30.0)
    bins = np.linspace(0.0, upper, 31)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method, durations in method_durations.items():
        if durations.size:
            axis.hist(durations, bins=bins, density=True, histtype="step", linewidth=2, label=method)
    axis.set_xlabel("Blink duration (seconds)")
    axis.set_ylabel("Density")
    axis.set_title("Aggregate blink-duration distribution")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_examples_plot(path, examples, start_threshold, end_threshold, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not examples:
        return
    figure, axes = plt.subplots(len(examples), 1, figsize=(12, 3 * len(examples)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (sequence_id, closure, events) in zip(axes, examples):
        time_axis = np.arange(len(closure)) / fps
        axis.plot(time_axis, closure, linewidth=1.2, label="Normalized closure")
        axis.axhline(start_threshold, color="tab:red", linestyle="--", label="Start threshold")
        axis.axhline(end_threshold, color="tab:green", linestyle=":", label="End threshold")
        for event in events:
            axis.axvspan(event["start_frame"] / fps, (event["end_frame"] + 1) / fps, color="tab:orange", alpha=0.25)
        axis.set_title(sequence_id)
        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Closure")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    mean, std = coefficient_transform(args.normalization, args.mean_face, args.std_face)

    train_files = coefficient_files(data_dir / "train" / "coefficients")
    calibration = calibrate_thresholds(
        train_files,
        tuple(args.blink_indices),
        mean,
        std,
        args.low_percentile,
        args.high_percentile,
        args.calibration_stride,
        args.max_calibration_frames,
        args.seed,
    )
    calibration.update({
        "blink_indices": args.blink_indices,
        "normalization": args.normalization,
        "start_threshold": args.start_threshold,
        "end_threshold": args.end_threshold,
        "merge_gap_frames": args.merge_gap_frames,
        "min_event_frames": args.min_event_frames,
        "fps": args.fps,
    })

    test_files = coefficient_files(data_dir / "test" / "coefficients")
    ground_truth_sequences = (
        (str(path.relative_to(data_dir / "test" / "coefficients")), np.load(path, mmap_mode="r"))
        for path in test_files
    )
    gt_summary, gt_rows, gt_durations, _ = analyze_sequences(
        ground_truth_sequences,
        "Ground Truth",
        calibration,
        tuple(args.blink_indices),
        args.fps,
        args.start_threshold,
        args.end_threshold,
        args.merge_gap_frames,
        args.min_event_frames,
        mean=mean,
        std=std,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        example_limit=0,
    )

    method_inputs = parse_method_inputs(args.input, args.results, args.method)
    raw_methods = set(args.raw_method)
    unknown_raw_methods = raw_methods.difference(method for method, _ in method_inputs)
    if unknown_raw_methods:
        raise ValueError(
            "--raw_method names without a matching --input: "
            + ", ".join(sorted(unknown_raw_methods))
        )

    summaries = [gt_summary]
    event_rows = list(gt_rows)
    duration_groups = {"Ground Truth": gt_durations}
    all_examples = []
    for method_index, (method, results_path) in enumerate(method_inputs):
        predictions = load_results(results_path, args.prediction_key)
        prediction_is_raw = args.prediction_space == "raw" or method in raw_methods
        prediction_mean = mean if prediction_is_raw else None
        prediction_std = std if prediction_is_raw else None
        pred_summary, pred_rows, pred_durations, examples = analyze_sequences(
            iter_nested_sequences(predictions, method),
            method,
            calibration,
            tuple(args.blink_indices),
            args.fps,
            args.start_threshold,
            args.end_threshold,
            args.merge_gap_frames,
            args.min_event_frames,
            mean=prediction_mean,
            std=prediction_std,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + method_index + 1,
            example_limit=args.example_count,
        )
        summaries.append(pred_summary)
        event_rows.extend(pred_rows)
        duration_groups[method] = pred_durations
        all_examples.extend(examples)

    with (output_dir / "blink_statistics.json").open("w", encoding="utf-8") as file:
        json.dump({"calibration": calibration, "methods": summaries}, file, indent=2, ensure_ascii=False)
    save_summary_csv(output_dir / "blink_summary.csv", summaries)
    save_events_csv(output_dir / "blink_events.csv", event_rows)
    save_distribution_plot(
        output_dir / "blink_duration_distribution.png",
        duration_groups,
    )
    save_examples_plot(
        output_dir / "blink_detection_examples.png",
        all_examples,
        args.start_threshold,
        args.end_threshold,
        args.fps,
    )

    for summary in summaries:
        ci = summary["mean_duration_ci95"]
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A"
        print(
            f"{summary['method']}: events={summary['events']}, "
            f"mean={summary['mean_duration_seconds']}, CI95={ci_text}, "
            f"median={summary['median_duration_seconds']}, "
            f"p95={summary['p95_duration_seconds']}, "
            f"blinks/min={summary['blinks_per_minute']}"
        )
    print(f"Saved blink analysis to {output_dir.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate 3DMM blink-duration statistics")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="METHOD=RESULTS_PT",
        help="Method name and results.pt path; repeat for multiple methods",
    )
    parser.add_argument("--results", help="Legacy single-method results.pt path")
    parser.add_argument("--method", default="FlowREACT")
    parser.add_argument("--prediction_key", default="PRED_3DMM")
    parser.add_argument("--prediction_space", choices=("normalized", "raw"), default="normalized")
    parser.add_argument(
        "--raw_method",
        action="append",
        default=[],
        metavar="METHOD",
        help="Treat this method's prediction as raw 3DMM; repeat when needed",
    )
    parser.add_argument("--normalization", choices=("none", "zero_center", "standard"), default="zero_center")
    parser.add_argument("--mean_face", default="external/FaceVerse/mean_face.npy")
    parser.add_argument("--std_face", default="external/FaceVerse/std_face.npy")
    parser.add_argument("--blink_indices", nargs=2, type=int, default=(8, 9))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--low_percentile", type=float, default=5.0)
    parser.add_argument("--high_percentile", type=float, default=95.0)
    parser.add_argument("--start_threshold", type=float, default=0.5)
    parser.add_argument("--end_threshold", type=float, default=0.2)
    parser.add_argument("--merge_gap_frames", type=int, default=2)
    parser.add_argument("--min_event_frames", type=int, default=2)
    parser.add_argument("--calibration_stride", type=int, default=10)
    parser.add_argument("--max_calibration_frames", type=int, default=2000000)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--example_count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", default="results/blink_duration")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
