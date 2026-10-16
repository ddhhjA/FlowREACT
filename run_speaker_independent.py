"""Run FlowREACT only, with isolated identity or recording partitions."""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from framework.utils.speaker_split import file_digest, load_split_bundle, protocol_partitions, verify_converter


REPO = Path(__file__).resolve().parent
METRICS = {"FRCorr": "FRC", "FRDist": "FRD", "FRDiv": "smse", "FRVar": "FRVar", "FRSyn": "TLCC"}


def hydra_value(value):
    # Quoted literals protect spaces/commas/equal signs without invoking a shell.
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def main_command(overrides):
    return [sys.executable, str(REPO / "main.py")] + [
        "{}={}".format(key, hydra_value(value)) for key, value in overrides.items()
    ]


def execute(command, log_file, dry_run=False):
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(REPO), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def mean_sd_ci(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    # Exact two-sided Student-t critical value for five independent training runs.
    margin = 2.7764451051977987 * sd / math.sqrt(5) if len(values) == 5 else None
    return {"mean": mean, "std": sd, "ci95": [mean - margin, mean + margin] if margin is not None else None}


def export_recording_table_row(output_dir, seed, group):
    metrics = group["per_seed"][str(seed)]
    row = {"condition": "Held-out recordings (retrained)", "FRC": metrics["FRC"],
           "FRD": metrics["FRD"], "FRVar": metrics["FRVar"], "FRSyn": metrics["TLCC"],
           "FRDiv": metrics["smse"], "num_samples": group["num_samples"]}
    with (output_dir / "recording_table_row.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    formatted = [row["condition"]] + ["{:.6f}".format(row[name])
                                      for name in ("FRC", "FRD", "FRVar", "FRSyn", "FRDiv")]
    formatted.append(str(row["num_samples"]))
    text = "| condition | FRC | FRD | FRVar | FRSyn | FRDiv | num_samples |\n"
    text += "|---|---:|---:|---:|---:|---:|---:|\n| " + " | ".join(formatted) + " |\n\n"
    text += ("Single training run (seed={}); retrained on a recording-disjoint split. "
             "This row uses a different training/test split from the original Clean/perturbation rows; "
             "it is not verified unseen-person evaluation.\n").format(seed)
    (output_dir / "recording_table_row.md").write_text(text, encoding="utf-8")
    return row


def summarize(output_dir, seeds, protocol="speaker_independent"):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct, nonempty training seeds")
    output_dir = Path(output_dir)
    evaluation_groups = protocol_partitions(protocol)[2:]
    recording_protocol = protocol == "recording_disjoint"
    groups = {}
    split_ids = set()
    training_runs = {}
    for group in evaluation_groups:
        payloads = []
        for seed in seeds:
            path = output_dir / group / ("seed_{}".format(seed)) / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            evaluation = payload["evaluation"]
            if (evaluation.get("protocol") != protocol or evaluation.get("group") != group
                    or payload["seed"] != seed or payload["num_preds"] != 10):
                raise ValueError("Unexpected evaluation protocol/seed/group at {}".format(path))
            if recording_protocol and (evaluation.get("training_seed") != seed
                                       or evaluation["counts"]["samples"] != payload["num_samples"]):
                raise ValueError("Training seed or evaluated sample count mismatch at {}".format(path))
            split_ids.add(evaluation["split_id"])
            run = evaluation["training_run_id"]
            if group == evaluation_groups[0]:
                if run in training_runs.values():
                    raise ValueError("Repeated inference of one checkpoint is not independent training")
                training_runs[seed] = run
            elif run != training_runs[seed]:
                raise ValueError("Evaluation groups must use the same trained model for each seed")
            for raw_key in METRICS.values():
                if not math.isfinite(payload["metrics"][raw_key]):
                    raise ValueError("Nonfinite metric at {}".format(path))
            payloads.append(payload)
        if len({p["num_samples"] for p in payloads}) != 1:
            raise ValueError("All seeds must evaluate the same group size")
        if len({p["evaluation"].get("converter_sha256") for p in payloads}) != 1:
            raise ValueError("All seeds must use the same isolated, frozen converter")
        groups[group] = {
            "num_seeds": len(seeds), "num_samples": payloads[0]["num_samples"],
            "num_speakers": payloads[0]["evaluation"]["counts"]["speakers"],
            "metrics": {name: mean_sd_ci([p["metrics"][key] for p in payloads])
                        for name, key in METRICS.items()},
            "per_seed": {str(seed): p["metrics"] for seed, p in zip(seeds, payloads)},
        }
        if recording_protocol:
            if groups[group]["num_speakers"] is not None:
                raise ValueError("Recording-only evaluation cannot claim participant counts")
            groups[group]["num_recordings"] = payloads[0]["evaluation"]["counts"]["recordings"]
            if len({p["evaluation"]["counts"]["recordings"] for p in payloads}) != 1:
                raise ValueError("All seeds must evaluate the same held-out recordings")
    if len(split_ids) != 1:
        raise ValueError("Cannot pool experiments with different data splits")
    result = {"method": "FlowREACT", "protocol": protocol, "split_id": next(iter(split_ids)),
              "training_seeds": seeds, "groups": groups,
              "ci95_scope": ("Student-t interval across five training seeds; not across participant identities"
                             if len(seeds) == 5 else None)}
    if recording_protocol:
        result["scope"] = "Held-out recordings only; participant identities are unavailable."
        if len(seeds) == 1:
            result["statistics_note"] = "Single training run; no across-run standard deviation or confidence interval."
            result["table_row"] = export_recording_table_row(output_dir, seeds[0], groups["heldout"])
    else:
        result["unseen_minus_seen"] = {
            name: mean_sd_ci([groups["unseen"]["per_seed"][str(seed)][key]
                              - groups["seen"]["per_seed"][str(seed)][key] for seed in seeds])
            for name, key in METRICS.items()
        }
        result["comparison_note"] = "Seen/unseen groups contain different recordings; differences can reflect content composition."
    summary_name = "recording_summary" if recording_protocol else "speaker_summary"
    (output_dir / (summary_name + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = ["method", "group", "num_seeds", "num_samples", "num_speakers"]
    if recording_protocol:
        fields.append("num_recordings")
    for name in METRICS:
        fields.extend([name + "_mean", name + "_std", name + "_ci95_low", name + "_ci95_high"])
    with (output_dir / (summary_name + ".csv")).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, data in groups.items():
            row = {"method": "FlowREACT", "group": group,
                   **{key: data[key] for key in ("num_seeds", "num_samples", "num_speakers")}}
            if recording_protocol:
                row["num_recordings"] = data["num_recordings"]
            for name, stats in data["metrics"].items():
                row.update({name + "_mean": stats["mean"], name + "_std": stats["std"],
                            name + "_ci95_low": stats["ci95"][0] if stats["ci95"] else "",
                            name + "_ci95_high": stats["ci95"][1] if stats["ci95"] else ""})
            writer.writerow(row)
    return result


def run(args):
    output = Path(args.output_dir).resolve()
    protocol = getattr(args, "protocol", "speaker_independent")
    evaluation_groups = protocol_partitions(protocol)[2:]
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be distinct and nonempty")
    if args.stage == "summarize":
        print(json.dumps(summarize(output, args.seeds, protocol), indent=2))
        return
    if not args.data_dir or not args.split_dir:
        raise ValueError("--data_dir and --split_dir are required")
    if (args.window_size <= 0 or args.clip_length <= 0 or args.clip_length % args.window_size
            or args.epochs <= 0 or args.converter_epochs <= 0 or args.batch_size <= 0
            or args.workers < 0 or args.metric_threads <= 0 or not 0 <= args.history_weight <= 1):
        raise ValueError("Invalid experiment settings; clip_length must be divisible by window_size")
    split_dir = Path(args.split_dir).resolve()
    _, audit = load_split_bundle(split_dir)
    if audit["protocol"] != protocol:
        raise ValueError("Runner protocol does not match split.json")
    experiment_config = "motion_flow_fft_recording" if protocol == "recording_disjoint" else "motion_flow_fft_speaker"
    converter_run = "converter_{}".format(audit["split_id"][:12])
    converter_checkpoint = (Path(args.converter_checkpoint).resolve() if args.converter_checkpoint else
                            output / "checkpoints" / "converter" / converter_run
                            / "Emotion3DMMTransformer" / "checkpoint_best.pth")
    common = {
        "data": experiment_config, "data_dir": Path(args.data_dir).resolve(),
        "data.split_dir": split_dir, "data.workers": args.workers,
        "task": "online", "trainer.max_seq_len": args.clip_length,
        "trainer.window_size": args.window_size, "trainer.data_transform": "zero_center",
        "trainer.renderer.do_render": False,
    }
    config = {"split_id": audit["split_id"], "data_dir": str(Path(args.data_dir).resolve()), "training_seeds": args.seeds,
              "clip_length": args.clip_length, "window_size": args.window_size,
              "epochs": args.epochs, "converter_epochs": args.converter_epochs,
              "batch_size": args.batch_size, "history_mode": args.history_mode,
              "history_weight": args.history_weight, "converter_checkpoint": str(converter_checkpoint)}
    if protocol == "recording_disjoint":
        config["protocol"] = protocol
    manifest = output / "experiment.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Settings differ from experiment.json; use a new output_dir or matching arguments")
    elif not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.stage in {"all", "converter"} and not args.converter_checkpoint:
        if converter_checkpoint.exists():
            verify_converter(converter_checkpoint, split_dir)
            print("Using the existing isolated converter: {}".format(converter_checkpoint))
        else:
            overrides = dict(common, trainer="motion_transfer", stage="fit", seed=1234,
                             run_id=converter_run)
            overrides.update({
                "trainer.epochs": args.converter_epochs, "trainer.batch_size": args.batch_size,
                "trainer.val_interval": min(20, args.converter_epochs),
                "+trainer.validation_window_size": args.window_size,
                "trainer.ckpt_dir": output / "checkpoints" / "converter",
                "hydra.run.dir": output / "hydra" / converter_run,
            })
            execute(main_command(overrides), output / "logs" / "converter.log", args.dry_run)
            if not args.dry_run:
                if not converter_checkpoint.is_file():
                    raise FileNotFoundError("Converter training did not produce a best checkpoint")
                proof = {"split_id": audit["split_id"], "checkpoint_sha256": file_digest(converter_checkpoint),
                         "training_partition": "train", "training_run_id": converter_run,
                         "protocol": protocol}
                Path(str(converter_checkpoint) + ".speaker_split.json").write_text(
                    json.dumps(proof, indent=2), encoding="utf-8")
    if not args.dry_run:
        verify_converter(converter_checkpoint, split_dir)
    if args.stage == "converter":
        return

    for seed in args.seeds:
        run_id = "flowreact_seed_{}".format(seed)
        checkpoint = output / "checkpoints" / "flowreact" / run_id / "FlowReact" / "checkpoint_best.pth"
        overrides = dict(common, trainer=experiment_config, seed=seed, run_id=run_id)
        overrides.update({
            "trainer.model.emotion_transfer_ckpt": converter_checkpoint,
            "trainer.batch_size": args.batch_size, "trainer.epochs": args.epochs,
            "trainer.val_interval": min(5, args.epochs),
            "trainer.inference_history": args.history_mode, "trainer.history_weight": args.history_weight,
            "trainer.speaker_independent.output_dir": output,
            "trainer.ckpt_dir": output / "checkpoints" / "flowreact",
        })
        if args.stage in {"all", "train"}:
            if checkpoint.exists() and not args.dry_run:
                raise FileExistsError("Will not overwrite an independent training run: {}".format(checkpoint))
            train_overrides = dict(overrides, stage="fit")
            train_overrides["hydra.run.dir"] = output / "hydra" / run_id
            execute(main_command(train_overrides), output / "logs" / (run_id + ".log"), args.dry_run)
        if args.stage in {"all", "test"}:
            if not checkpoint.is_file() and not args.dry_run:
                raise FileNotFoundError("Train this seed first: {}".format(checkpoint))
            for group in evaluation_groups:
                test_overrides = dict(overrides, stage="test")
                test_overrides.update({
                    "data.evaluation_group": group, "data.test_dataset.batch_size": 1,
                    "trainer.speaker_independent.checkpoint": checkpoint,
                    "trainer.robustness.metric_threads": args.metric_threads,
                    "hydra.run.dir": output / "hydra" / (run_id + "_" + group),
                })
                metrics_path = output / group / ("seed_{}".format(seed)) / "metrics.json"
                if metrics_path.exists() and not args.dry_run:
                    raise FileExistsError("Will not overwrite saved evaluation: {}".format(metrics_path))
                execute(main_command(test_overrides), output / "logs" / (run_id + "_" + group + ".log"), args.dry_run)
    if args.stage in {"all", "test"} and not args.dry_run:
        print(json.dumps(summarize(output, args.seeds, protocol), indent=2))


def main(protocol="speaker_independent"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "converter", "train", "test", "summarize"), default="all")
    parser.add_argument("--data_dir")
    parser.add_argument("--split_dir")
    parser.set_defaults(protocol=protocol)
    parser.add_argument("--output_dir", default="results/recording_independent" if protocol == "recording_disjoint"
                        else "results/speaker_independent")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[1234] if protocol == "recording_disjoint" else [1234, 1235, 1236, 1237, 1238])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--converter_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--clip_length", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metric_threads", type=int, default=4)
    parser.add_argument("--history_mode", choices=("previous_window", "legacy_zero"), default="previous_window")
    parser.add_argument("--history_weight", type=float, default=1.0)
    parser.add_argument("--converter_checkpoint")
    parser.add_argument("--dry_run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
