"""Exercise the actual dataset methods without unrelated video/audio import dependencies."""

import ast
import csv
import os
import random
import tempfile
import types
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

from framework.utils.speaker_split import IDENTITY_FIELDS, prepare_recording_split, prepare_split, write_csv


@unittest.skipIf(torch is None, "CPU tensor checks require PyTorch and NumPy")
class SpeakerDatasetCPUTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rows = []
        for speaker in range(4):
            for recording in range(4):
                name = "recording_{}_{}".format(speaker, recording)
                rows.append({"source_split": "train", "recording_id": name,
                             "speaker_id": "person_{}".format(speaker),
                             "listener_id": "listener_{}_{}".format(speaker, recording)})
                for role in ("speaker", "listener"):
                    for modality, dimension in (("coefficients", 58), ("facial-attributes", 25),
                                                ("audio-features", 768)):
                        path = self.root / "train" / modality / role / "session0" / (name + ".npy")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shape = (32, 1, dimension) if modality == "coefficients" else (32, dimension)
                        np.save(str(path), np.ones(shape, dtype=np.float32) * (speaker + 1))
        identity_file = self.root / "identities.csv"
        write_csv(identity_file, IDENTITY_FIELDS, rows)
        self.split_dir = self.root / "split"
        prepare_split(self.root, identity_file, self.split_dir, unseen_speakers=["person_3"])
        face = self.root / "external" / "FaceVerse"
        face.mkdir(parents=True)
        np.save(str(face / "mean_face.npy"), np.zeros(58, dtype=np.float32))
        np.save(str(face / "std_face.npy"), np.ones(58, dtype=np.float32))

        source = Path(__file__).resolve().parents[1] / "dataset" / "react_2025.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in {"ReactionDataset", "custom_collate"}]
        module = ast.Module(body=definitions, type_ignores=[])
        def forbid_video(*args, **kwargs):
            raise AssertionError("Manifest tensor-only experiments must not decode video")
        hydra_utils = types.SimpleNamespace(get_original_cwd=lambda: str(self.root),
                                           to_absolute_path=lambda path: str(Path(path).resolve()))
        namespace = {"os": os, "Path": Path, "np": np, "torch": torch, "random": random,
                     "data": torch.utils.data, "hydra": types.SimpleNamespace(utils=hydra_utils),
                     "transforms": types.SimpleNamespace(Lambda=lambda function: function),
                     "Transform": lambda *args: lambda image: image,
                     "VideoReader": forbid_video, "cpu": lambda value: value}
        exec(compile(module, str(source), "exec"), namespace)
        self.Dataset = namespace["ReactionDataset"]
        self.collate = namespace["custom_collate"]

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, partition, **kwargs):
        return self.Dataset(root_dir=str(self.root), split="test" if partition in {"seen", "unseen"} else partition,
                            pair_manifest=str(self.split_dir / (partition + ".csv")), clip_length=16,
                            load_video_s=False, load_video_l=False, normalize_3dmm="zero_center", **kwargs)

    def test_training_and_development_shapes_and_real_inputs(self):
        for partition in ("train", "val"):
            dataset = self.dataset(partition)
            item = dataset[0]
            self.assertEqual(tuple(item[0].shape), (16, 768))
            self.assertEqual(tuple(item[2].shape), (16, 25))
            self.assertEqual(tuple(item[3].shape), (16, 58))
            self.assertEqual(tuple(item[5].shape), (16, 25))
            self.assertEqual(tuple(item[6].shape), (16, 58))
            self.assertGreater(torch.count_nonzero(item[0]).item(), 0)
            self.assertGreater(torch.count_nonzero(item[3]).item(), 0)
            self.assertEqual(len(self.collate([item, item])[0]), 2)

    def test_full_length_evaluation_and_reproducible_gt_candidates(self):
        for partition in ("seen", "unseen"):
            dataset = self.dataset(partition)
            random.seed(1)
            first = dataset[0]
            random.seed(999)
            second = dataset[0]
            self.assertEqual(tuple(first[0].shape), (32, 768))
            self.assertEqual(tuple(first[3].shape), (32, 58))
            self.assertEqual(int(first[7]), 32)
            self.assertEqual(len(first[4]), 10)
            self.assertEqual(len(first[5]), 10)
            self.assertEqual(len(first[6]), 10)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(first[5], second[5])))

    def test_manifest_mode_cannot_be_mislabelled_as_training(self):
        with self.assertRaisesRegex(ValueError, "require split=test"):
            self.Dataset(root_dir=str(self.root), split="train",
                         pair_manifest=str(self.split_dir / "unseen.csv"))

    def test_recording_heldout_dataset_uses_full_length_nonzero_features(self):
        recording_dir = self.root / "recording_split"
        prepare_recording_split(self.root, recording_dir)
        for partition in ("train", "val", "heldout"):
            dataset = self.Dataset(root_dir=str(self.root), split="test" if partition == "heldout" else partition,
                                   pair_manifest=str(recording_dir / (partition + ".csv")), clip_length=16,
                                   load_video_s=False, load_video_l=False, normalize_3dmm="zero_center")
            item = dataset[0]
            length = 32 if partition == "heldout" else 16
            self.assertEqual(tuple(item[0].shape), (length, 768))
            self.assertEqual(tuple(item[3].shape), (length, 58))
            self.assertGreater(torch.count_nonzero(item[0]).item(), 0)
            self.assertGreater(torch.count_nonzero(item[3]).item(), 0)
            self.assertEqual(dataset.identity_protocol["protocol"], "recording_disjoint")
            self.assertNotIn("speaker_id", dataset.sample_metadata[0])
            if partition == "heldout":
                self.assertEqual(len(item[5]), 10)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(item[5], dataset[0][5])))

    def test_legacy_discovery_still_loads_both_roles(self):
        for role in ("speaker", "listener"):
            path = self.root / "train" / "video-face-crop" / role / "session0" / "legacy.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not decoded by initialization")
        dataset = self.Dataset(root_dir=str(self.root), clip_length=16)
        self.assertEqual(len(dataset), 2)
        self.assertIsNone(dataset.identity_protocol)
        self.assertTrue(all(len(path.parts) == 3 for path in dataset.speaker_path_list))


if __name__ == "__main__":
    unittest.main()
