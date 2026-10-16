import copy
import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from framework.utils.speaker_split import (
    load_split_bundle, manifest_dataset_paths, prepare_recording_split,
    validate_recording_partitions, verify_converter, file_digest,
)
from run_speaker_independent import main, run, summarize


class RecordingIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        # Each recording has different sessions in both physical source archives.
        for index in range(20):
            for source, session in (("train", "session0"), ("val", "session1")):
                self.add_pair(source, session, "camera_{}".format(index))

    def add_pair(self, source, session, recording):
        for role in ("speaker", "listener"):
            for modality in ("coefficients", "facial-attributes", "audio-features"):
                path = self.data / source / modality / role / session / (recording + ".npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic feature placeholder")

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, destination="split", **kwargs):
        output = self.root / destination
        return output, prepare_recording_split(self.data, output, **kwargs)

    def args(self, split_dir):
        return Namespace(output_dir=str(self.root / "outputs"), seeds=[1234], stage="all",
                         data_dir=str(self.data), split_dir=str(split_dir), window_size=8,
                         clip_length=256, epochs=1, converter_epochs=1, batch_size=2,
                         workers=0, metric_threads=1, history_weight=1.0,
                         history_mode="previous_window", converter_checkpoint=None, dry_run=True,
                         protocol="recording_disjoint")

    def test_all_sessions_and_directions_stay_in_one_group_without_person_ids(self):
        split, audit = self.prepare()
        partitions, loaded = load_split_bundle(split)
        self.assertEqual(loaded, audit)
        self.assertEqual(audit["protocol"], "recording_disjoint")
        self.assertIsNone(audit["speaker_identity_disjoint"])
        self.assertEqual([audit["counts"][p]["recordings"] for p in ("train", "val", "heldout")], [13, 3, 4])
        self.assertEqual(sum(len(rows) for rows in partitions.values()), 80)
        for rows in partitions.values():
            self.assertEqual({r["source_split"] for r in rows}, {"train", "val"})
            self.assertTrue(all("speaker_id" not in row and "listener_id" not in row for row in rows))
        self.assertTrue(all(c["speakers"] is None for c in audit["counts"].values()))

    def test_fixed_seed_is_reproducible_and_output_is_immutable(self):
        split, first = self.prepare("a")
        _, second = self.prepare("b")
        _, changed = self.prepare("c", seed=4321)
        self.assertEqual(first["split_id"], second["split_id"])
        self.assertNotEqual(first["split_id"], changed["split_id"])
        with self.assertRaises(FileExistsError):
            prepare_recording_split(self.data, split)

    def test_duplicate_archived_session_is_rejected(self):
        self.add_pair("val", "session0", "camera_0")
        with self.assertRaisesRegex(ValueError, "duplicated across source"):
            self.prepare()

    def test_leakage_and_missing_reverse_direction_are_rejected(self):
        split, _ = self.prepare()
        partitions, _ = load_split_bundle(split)
        leaked = copy.deepcopy(partitions)
        leaked["train"].extend(copy.deepcopy(leaked["heldout"][:2]))
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_recording_partitions(leaked)
        missing = copy.deepcopy(partitions)
        missing["heldout"].pop(0)
        with self.assertRaisesRegex(ValueError, "both input directions"):
            validate_recording_partitions(missing)

    def test_path_traversal_and_missing_features_are_rejected(self):
        split, _ = self.prepare()
        partitions, _ = load_split_bundle(split)
        partitions["train"][0]["speaker_path"] = "../session0/escape"
        with self.assertRaisesRegex(ValueError, "Expected role"):
            validate_recording_partitions(partitions)
        next((self.data / "val" / "audio-features").glob("*/*/*.npy")).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "required feature"):
            self.prepare("missing")

    def test_hashes_counts_and_assignment_metadata_are_audited(self):
        split, audit = self.prepare()
        audit["counts"]["train"]["samples"] += 1
        (split / "split.json").write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "counts"):
            load_split_bundle(split)
        split, audit = self.prepare("second")
        audit["recording_groups"]["train"].append("made_up")
        (split / "split.json").write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "assignment metadata"):
            load_split_bundle(split)
        split, _ = self.prepare("third")
        with (split / "heldout.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            load_split_bundle(split)

    def test_evaluation_gt_pool_excludes_train_and_development(self):
        split, _ = self.prepare()
        speakers, listeners, ground_truths, rows, metadata = manifest_dataset_paths(split / "heldout.csv")
        self.assertEqual(metadata["partition"], "heldout")
        self.assertEqual(len(speakers), len(rows))
        self.assertTrue(all(listener in candidates and set(candidates) <= set(listeners)
                            for listener, candidates in zip(listeners, ground_truths)))

    def test_converter_provenance_checks_work_for_recording_protocol(self):
        split, audit = self.prepare()
        checkpoint = self.root / "converter.pth"
        checkpoint.write_bytes(b"converter from isolated training recordings")
        with self.assertRaises(FileNotFoundError):
            verify_converter(checkpoint, split)
        Path(str(checkpoint) + ".speaker_split.json").write_text(json.dumps({
            "protocol": "recording_disjoint", "split_id": audit["split_id"],
            "training_partition": "train", "checkpoint_sha256": file_digest(checkpoint),
        }))
        verify_converter(checkpoint, split)

    def test_runner_only_schedules_converter_flow_training_and_heldout_test(self):
        split, _ = self.prepare()
        commands = []
        with patch("run_speaker_independent.execute", side_effect=lambda command, *a: commands.append(command)):
            run(self.args(split))
        self.assertEqual(len(commands), 3)
        self.assertIn('trainer="motion_transfer"', commands[0])
        self.assertIn('trainer="motion_flow_fft_recording"', commands[1])
        self.assertIn('data.evaluation_group="heldout"', commands[2])
        self.assertFalse((self.root / "outputs").exists())

    def test_runner_rejects_identity_protocol_mislabelling(self):
        split, _ = self.prepare()
        args = self.args(split)
        args.protocol = "speaker_independent"
        with self.assertRaisesRegex(ValueError, "Runner protocol"):
            run(args)

    def test_recording_entrypoint_defaults_to_one_seed(self):
        with patch("sys.argv", ["run_recording_independent.py"]), patch("run_speaker_independent.run") as runner:
            main(protocol="recording_disjoint")
        self.assertEqual(runner.call_args[0][0].seeds, [1234])

    def test_single_run_exports_exact_table_row_without_variability_claims(self):
        path = self.root / "outputs" / "heldout" / "seed_1234" / "metrics.json"
        path.parent.mkdir(parents=True)
        synthetic = {"FRC": 0.21, "FRD": 160.1, "smse": 0.12, "FRVar": 0.08, "TLCC": 48.5}
        path.write_text(json.dumps({
            "seed": 1234, "num_preds": 10, "num_samples": 16,
            "evaluation": {"protocol": "recording_disjoint", "group": "heldout", "split_id": "split",
                           "training_seed": 1234, "training_run_id": "flow_1234", "converter_sha256": "converter",
                           "counts": {"samples": 16, "recordings": 4, "speakers": None}},
            "metrics": synthetic,
        }))
        result = summarize(self.root / "outputs", [1234], "recording_disjoint")
        self.assertIsNone(result["ci95_scope"])
        self.assertTrue(all(s["std"] is None and s["ci95"] is None
                            for s in result["groups"]["heldout"]["metrics"].values()))
        with (self.root / "outputs" / "recording_table_row.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["FRSyn"]), synthetic["TLCC"])
        self.assertEqual(float(rows[0]["FRDiv"]), synthetic["smse"])
        self.assertIn("different training/test split", (self.root / "outputs" / "recording_table_row.md").read_text())

    def test_summary_has_only_real_recording_counts_and_five_training_runs(self):
        seeds = [1234, 1235, 1236, 1237, 1238]
        for index, seed in enumerate(seeds):
            path = self.root / "outputs" / "heldout" / ("seed_{}".format(seed)) / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "seed": seed, "num_preds": 10, "num_samples": 16,
                "evaluation": {"protocol": "recording_disjoint", "group": "heldout", "split_id": "split",
                               "training_seed": seed,
                               "training_run_id": "flow_{}".format(seed), "converter_sha256": "converter",
                               "counts": {"samples": 16, "recordings": 4, "speakers": None}},
                "metrics": {key: 0.1 + index * 0.001 for key in ("FRC", "FRD", "smse", "FRVar", "TLCC")},
            }))
        result = summarize(self.root / "outputs", seeds, "recording_disjoint")
        self.assertEqual(list(result["groups"]), ["heldout"])
        self.assertIsNone(result["groups"]["heldout"]["num_speakers"])
        self.assertEqual(result["groups"]["heldout"]["num_recordings"], 4)
        self.assertNotIn("unseen_minus_seen", result)
        self.assertIsNotNone(result["groups"]["heldout"]["metrics"]["FRCorr"]["ci95"])
        with (self.root / "outputs" / "recording_summary.csv").open(newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["num_speakers"], "")


if __name__ == "__main__":
    unittest.main()
