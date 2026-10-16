import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from framework.utils.speaker_split import (
    IDENTITY_FIELDS, discover_recordings, file_digest, identity_template,
    load_split_bundle, manifest_dataset_paths, prepare_split, validate_partitions,
    verify_converter, write_csv,
)
from run_speaker_independent import main_command, mean_sd_ci, summarize


class SpeakerIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.identities = self.root / "identities.csv"
        rows = []
        for speaker in range(5):
            for recording in range(4):
                name = "camera_{}_{}".format(speaker, recording)
                rows.append({"source_split": "train", "recording_id": name,
                             "speaker_id": "person_{}".format(speaker),
                             "listener_id": "listener_{}_{}".format(speaker, recording)})
                for session in ("session0", "session1"):
                    for role in ("speaker", "listener"):
                        for modality in ("coefficients", "facial-attributes", "audio-features"):
                            path = self.data / "train" / modality / role / session / (name + ".npy")
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(b"synthetic feature placeholder")
        write_csv(self.identities, IDENTITY_FIELDS, rows)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, destination="split", **kwargs):
        output = self.root / destination
        audit = prepare_split(self.data, self.identities, output,
                              unseen_speakers=["person_4"], **kwargs)
        return output, audit

    def test_template_is_per_recording_not_per_session(self):
        template = self.root / "template.csv"
        self.assertEqual(identity_template(self.data, template), 20)
        with template.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(not row["speaker_id"] and not row["listener_id"] for row in rows))
        with self.assertRaisesRegex(ValueError, "Unfilled"):
            prepare_split(self.data, template, self.root / "bad")
        with self.assertRaises(FileExistsError):
            identity_template(self.data, template)

    def test_split_is_recording_disjoint_and_unseen_in_neither_role(self):
        output, audit = self.prepare()
        partitions, _ = load_split_bundle(output)
        for partition in ("train", "val"):
            all_ids = {row[key] for row in partitions[partition] for key in ("speaker_id", "listener_id")}
            self.assertNotIn("person_4", all_ids)
        self.assertEqual({row["speaker_id"] for row in partitions["unseen"]}, {"person_4"})
        self.assertFalse(audit["listener_identity_disjoint"])
        memberships = {}
        for name, rows in partitions.items():
            for row in rows:
                memberships.setdefault(row["recording_id"], set()).add(name)
        self.assertTrue(all(len(groups) == 1 for groups in memberships.values()))

    def test_held_out_person_as_listener_is_excluded_too(self):
        with self.identities.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["listener_id"] = "person_4"
        write_csv(self.identities, IDENTITY_FIELDS, rows)
        output, _ = self.prepare()
        partitions, _ = load_split_bundle(output)
        self.assertTrue(any(row["speaker_path"].startswith("listener/") for row in partitions["unseen"]))
        self.assertFalse(any(row["recording_id"] == "camera_0_0" for row in partitions["train"]))

    def test_split_is_reproducible_and_cannot_be_silently_overwritten(self):
        output, first = self.prepare("a")
        _, second = self.prepare("b")
        self.assertEqual(first["split_id"], second["split_id"])
        with self.assertRaises(FileExistsError):
            prepare_split(self.data, self.identities, output)

    def test_modified_manifest_is_rejected(self):
        output, _ = self.prepare()
        with (output / "train.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            load_split_bundle(output)

    def test_gt_candidates_are_only_from_evaluation_partition(self):
        output, _ = self.prepare()
        speakers, listeners, ground_truths, rows, metadata = manifest_dataset_paths(output / "unseen.csv")
        allowed = set(listeners)
        self.assertEqual(len(speakers), len(rows))
        self.assertEqual(metadata["partition"], "unseen")
        for listener, candidates in zip(listeners, ground_truths):
            self.assertIn(listener, candidates)
            self.assertTrue(set(candidates) <= allowed)
            self.assertTrue(all(path.parts[2] == listener.parts[2] for path in candidates))

    def test_validation_rejects_leakage_and_path_traversal(self):
        output, audit = self.prepare()
        partitions, _ = load_split_bundle(output)
        leaked = copy.deepcopy(partitions)
        leaked["train"].append(copy.deepcopy(leaked["unseen"][0]))
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_partitions(leaked, audit["unseen_speaker_ids"])
        bad = copy.deepcopy(partitions)
        bad["train"][0]["speaker_path"] = "../outside/session0"
        with self.assertRaisesRegex(ValueError, "Expected role"):
            validate_partitions(bad, audit["unseen_speaker_ids"])

    def test_missing_features_are_not_silently_skipped(self):
        first = next((self.data / "train" / "audio-features").glob("*/*/*.npy"))
        first.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "required feature"):
            self.prepare()

    def test_converter_must_have_matching_provenance(self):
        output, audit = self.prepare()
        checkpoint = self.root / "converter.pth"
        checkpoint.write_bytes(b"isolated pretrained converter")
        with self.assertRaises(FileNotFoundError):
            verify_converter(checkpoint, output)
        proof = Path(str(checkpoint) + ".speaker_split.json")
        proof.write_text(json.dumps({"split_id": audit["split_id"], "training_partition": "train",
                                     "checkpoint_sha256": file_digest(checkpoint)}))
        verify_converter(checkpoint, output)
        checkpoint.write_bytes(b"different converter")
        with self.assertRaises(ValueError):
            verify_converter(checkpoint, output)

    def test_hydra_command_uses_quoted_literals_and_no_shell(self):
        command = main_command({"data_dir": Path("data with spaces"), "resume": False, "seed": 1234})
        self.assertEqual(command[-3], 'data_dir="data with spaces"')
        self.assertEqual(command[-2:], ["resume=false", "seed=1234"])

    def test_five_seed_statistics_use_sample_sd(self):
        stats = mean_sd_ci([1, 2, 3, 4, 5])
        self.assertEqual(stats["mean"], 3)
        self.assertAlmostEqual(stats["std"], math.sqrt(2.5))
        self.assertAlmostEqual(stats["ci95"][1], 3 + 2.7764451051977987 * math.sqrt(0.5))
        self.assertIsNone(mean_sd_ci([1])["std"])
        self.assertIsNone(mean_sd_ci([1, 2])["ci95"])

    def create_metric_files(self):
        seeds = [1234, 1235, 1236, 1237, 1238]
        for group in ("seen", "unseen"):
            for i, seed in enumerate(seeds):
                path = self.root / "results" / group / ("seed_{}".format(seed)) / "metrics.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "seed": seed, "num_preds": 10, "num_samples": 8,
                    "evaluation": {"protocol": "speaker_independent", "group": group,
                                   "split_id": "one_split", "training_run_id": "training_{}".format(seed),
                                   "converter_sha256": "one_converter", "counts": {"speakers": 2}},
                    "metrics": {key: 1 + i * 0.1 for key in ("FRC", "FRD", "smse", "FRVar", "TLCC")},
                }))
        return seeds

    def test_summary_matches_paired_training_runs_and_aliases(self):
        seeds = self.create_metric_files()
        summary = summarize(self.root / "results", seeds)
        self.assertEqual(summary["method"], "FlowREACT")
        self.assertAlmostEqual(summary["groups"]["seen"]["metrics"]["FRDiv"]["mean"], 1.2)
        self.assertEqual(summary["unseen_minus_seen"]["FRCorr"]["mean"], 0)
        self.assertTrue((self.root / "results" / "speaker_summary.csv").exists())

    def test_summary_rejects_same_training_checkpoint_counted_twice(self):
        seeds = self.create_metric_files()
        path = self.root / "results" / "seen" / "seed_1235" / "metrics.json"
        payload = json.loads(path.read_text())
        payload["evaluation"]["training_run_id"] = "training_1234"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "independent training"):
            summarize(self.root / "results", seeds)


if __name__ == "__main__":
    unittest.main()
