"""Compute speaker-expression subset metrics from an existing clean test output."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select(items, indices):
    return [items[index] for index in indices]


def mean_valence(emotion, valid_length, valence_index):
    emotion = torch.as_tensor(emotion)
    if emotion.ndim != 2:
        raise ValueError(f"Expected speaker emotion [T, D], got {tuple(emotion.shape)}")
    if not 0 <= valence_index < emotion.shape[-1]:
        raise ValueError(
            f"valence_index={valence_index} is outside emotion dimension {emotion.shape[-1]}"
        )
    valid_length = max(1, min(int(valid_length), emotion.shape[0]))
    return float(emotion[:valid_length, valence_index].float().mean().item())


def build_valence_groups(speaker_emotions, predictions, threshold, valence_index):
    scores = []
    groups = {"negative": [], "neutral": [], "positive": []}
    for index, (emotion, prediction) in enumerate(zip(speaker_emotions, predictions)):
        valid_length = torch.as_tensor(prediction).shape[-2]
        score = mean_valence(emotion, valid_length, valence_index)
        scores.append(score)
        if score < -threshold:
            groups["negative"].append(index)
        elif score > threshold:
            groups["positive"].append(index)
        else:
            groups["neutral"].append(index)
    return np.asarray(scores, dtype=np.float64), groups


def build_activity_groups(activities, low_quantile, high_quantile):
    low_value, high_value = np.quantile(activities, [low_quantile, high_quantile])
    groups = {
        "low": np.flatnonzero(activities <= low_value).tolist(),
        "medium": np.flatnonzero(
            (activities > low_value) & (activities <= high_value)
        ).tolist(),
        "high": np.flatnonzero(activities > high_value).tolist(),
    }
    return (float(low_value), float(high_value)), groups


def main(args):
    from framework.utils.compute_metrics import compute_metrics

    input_path = Path(args.input)
    payload = load_payload(input_path)
    predictions = payload["PRED"]
    targets = payload["GT"]
    speaker_emotions = payload["SPEAKER_EMOTION"]

    expected = len(predictions)
    if not (len(targets) == len(speaker_emotions) == expected):
        raise RuntimeError(
            "Predictions, targets, and speaker inputs are not aligned: "
            f"{len(predictions)}, {len(targets)}, {len(speaker_emotions)}"
        )

    output = {
        "input": str(input_path.resolve()),
        "num_samples": expected,
        "group_by": args.group_by,
        "groups": {},
    }

    if args.group_by == "valence":
        scores, group_indices = build_valence_groups(
            speaker_emotions,
            predictions,
            threshold=args.neutral_threshold,
            valence_index=args.valence_index,
        )
        output["grouping"] = {
            "definition": "mean speaker valence over valid prediction frames",
            "valence_index": args.valence_index,
            "negative": f"mean_valence < {-args.neutral_threshold}",
            "neutral": (
                f"{-args.neutral_threshold} <= mean_valence <= {args.neutral_threshold}"
            ),
            "positive": f"mean_valence > {args.neutral_threshold}",
        }
    else:
        if "EXPRESSION_ACTIVITY" not in payload:
            raise KeyError(
                "The clean results do not contain EXPRESSION_ACTIVITY. "
                "Run with --group_by valence or regenerate clean results once."
            )
        scores = np.asarray(payload["EXPRESSION_ACTIVITY"], dtype=np.float64)
        if len(scores) != expected:
            raise RuntimeError(
                "EXPRESSION_ACTIVITY is not aligned with predictions: "
                f"{len(scores)} versus {expected}"
            )
        quantile_values, group_indices = build_activity_groups(
            scores,
            args.low_quantile,
            args.high_quantile,
        )
        output["grouping"] = {
            "definition": "mean temporal variance of the first 52 speaker 3DMM coefficients",
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
            "low_value": quantile_values[0],
            "high_value": quantile_values[1],
        }

    torch.multiprocessing.set_sharing_strategy("file_system")
    for group, indices in group_indices.items():
        if not indices:
            print(f"Skipping empty expression group: {group}")
            continue
        print(f"\nComputing {group} metrics for {len(indices)} samples")
        metrics = compute_metrics(
            select(speaker_emotions, indices),
            select(predictions, indices),
            select(targets, indices),
            threads=args.metric_threads,
        )
        group_scores = scores[indices]
        output["groups"][group] = {
            "num_samples": len(indices),
            "score_mean": float(group_scores.mean()),
            "score_sd": float(group_scores.std(ddof=1)) if len(indices) > 1 else 0.0,
            "metrics": {key: float(value) for key, value in metrics.items()},
        }

    output_path = Path(args.output) if args.output else input_path.with_name("expression_groups.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    print(f"\nSaved expression-group metrics to {output_path.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze FlowREACT metrics by speaker expression condition"
    )
    parser.add_argument("--input", required=True, help="Clean robustness results.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--group_by",
        choices=("valence", "activity"),
        default="valence",
        help="Use negative/neutral/positive valence or low/medium/high expression activity",
    )
    parser.add_argument("--valence_index", type=int, default=15)
    parser.add_argument("--neutral_threshold", type=float, default=0.1)
    parser.add_argument("--low_quantile", type=float, default=0.33)
    parser.add_argument("--high_quantile", type=float, default=0.67)
    parser.add_argument("--metric_threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
