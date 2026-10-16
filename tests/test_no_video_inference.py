"""Run the real inference/dataset methods without optional GPU/video dependencies."""

import ast
import json
import logging
import math
import os
import random
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative_path, names, namespace, methods=None):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if methods is not None:
        for node in definitions:
            node.bases = []
            node.body = [method for method in node.body
                         if isinstance(method, ast.FunctionDef) and method.name in methods]
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class NoVideoInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.model_calls = []
        self.metric_calls = []

        class Model:
            def eval(model):
                return model

            def __call__(model, motions, audios, **kwargs):
                self.model_calls.append((motions, audios, kwargs))
                shape = (len(motions), motions[0].shape[0])
                return {"listener_3dmm_out": torch.full((*shape, 58), 0.2),
                        "listener_emotion_out": torch.full((*shape, 25), 0.3)}

        class Processor:
            def __init__(processor, **kwargs):
                pass

            def forward(processor, prediction_list, target_list):
                return [torch.stack(targets) for targets in target_list]

        def metrics(speakers, predictions, targets, **kwargs):
            self.assertGreater(len(predictions), 0)
            self.assertEqual(len(speakers), len(predictions))
            self.assertEqual(len(targets), len(predictions))
            self.assertTrue(all(tuple(pred.shape) == (10, 8, 25) for pred in predictions))
            self.metric_calls.append((speakers, predictions, targets))
            return {"FRC": 0.25}

        namespace = {
            "torch": torch, "math": math, "json": json, "Path": Path, "os": os,
            "logger": logging.getLogger(__name__), "tqdm": lambda loader: loader,
            "hydra": types.SimpleNamespace(utils=types.SimpleNamespace(
                to_absolute_path=lambda path: str(Path(path).resolve()))),
            "instantiate": lambda *args, **kwargs: Model(),
            "from_pretrained_checkpoint": lambda *args: None,
            "Processor": Processor, "compute_metrics": metrics,
            "validate_robustness_mode": lambda mode: mode,
            "robustness_condition_name": lambda mode, snr: mode,
            "clone_tensor_list": lambda clips: [clip.clone() for clip in clips],
            "compute_expression_activity": lambda *args: 0.0,
            "apply_input_perturbation": lambda audio, motion, lengths, **kwargs: (audio, motion),
            "compute_input_audit": lambda *args, **kwargs: {},
        }
        self.Trainer = load_definitions("trainer/motion_flow_fft.py", {"Trainer"}, namespace,
                                        methods={"pad_to", "test"})["Trainer"]

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def batch(size=1):
        return ([torch.ones(8, 768) for _ in range(size)], [],
                [torch.ones(8, 25) for _ in range(size)],
                [torch.ones(8, 58) for _ in range(size)], [],
                [[torch.ones(8, 25) for _ in range(10)] for _ in range(size)],
                [[torch.ones(8, 58) for _ in range(10)] for _ in range(size)],
                [8] * size, [torch.tensor([8] * 10) for _ in range(size)])

    def trainer(self, batches):
        class Loader:
            batch_size = len(batches[0][7]) if batches else 1
            dataset = types.SimpleNamespace(sample_metadata=None)

            def __iter__(loader):
                return iter(batches)

        trainer = self.Trainer()
        trainer.device = torch.device("cpu")
        trainer.model_cfg = types.SimpleNamespace(task="online")
        trainer.speaker_independent_cfg = {}
        trainer.evaluation_cfg = {}
        trainer.robustness_cfg = {"enabled": True, "mode": "clean",
                                  "output_dir": str(self.output), "seed": 1234}
        trainer.renderer_cfg = {"do_render": False}
        trainer.kwargs = {"data_clamp": True, "post_config_name": "test", "post_clip_length": 8}
        trainer.window_size = trainer.max_seq_len = 8
        trainer.inference_history = "previous_window"
        trainer.history_weight = 1.0
        trainer.get_ckpt_path = lambda *args, **kwargs: "existing_checkpoint.pth"
        trainer.data_module = types.SimpleNamespace(get_dataloader=lambda **kwargs: Loader())
        return trainer

    def test_no_videos_still_generates_ten_candidates(self):
        self.trainer([self.batch()]).test()
        self.assertEqual(len(self.model_calls), 10)
        report = json.loads((self.output / "clean/seed_1234/metrics.json").read_text())
        self.assertEqual(report["num_samples"], 1)
        self.assertEqual(report["num_preds"], 10)
        self.assertEqual(len(self.metric_calls), 1)
        saved = torch.load(self.output / "clean/seed_1234/results.pt", weights_only=False)
        self.assertEqual(tuple(saved["PRED"][0].shape), (10, 8, 25))
        self.assertEqual(tuple(saved["PRED_3DMM"][0].shape), (10, 8, 58))

    def test_batch_of_two_is_not_truncated_by_optional_video_lists(self):
        self.trainer([self.batch(2)]).test()
        self.assertEqual(len(self.model_calls), 20)
        self.assertEqual(len(self.metric_calls[0][1]), 2)
        self.assertTrue(all(call[2]["history_mode"] == "previous_window" for call in self.model_calls))

    def test_missing_required_audio_fails_before_generation(self):
        batch = list(self.batch(2))
        batch[0].pop()
        with self.assertRaisesRegex(ValueError, "speaker_audio_clips has 1 samples; expected 2"):
            self.trainer([tuple(batch)]).test()
        self.assertEqual(self.model_calls, [])
        self.assertEqual(self.metric_calls, [])

    def test_empty_loader_fails_before_saving_or_metric_computation(self):
        with self.assertRaisesRegex(RuntimeError, "Inference produced no predictions"):
            self.trainer([]).test()
        self.assertEqual(self.metric_calls, [])
        self.assertFalse((self.output / "clean/seed_1234/results.pt").exists())

    def test_legacy_dataset_does_not_decode_disabled_listener_video(self):
        video_paths = []

        class MetadataOnlyVideo:
            def __init__(reader, path, **kwargs):
                video_paths.append(path)

            def __len__(reader):
                return 8

            def __getitem__(reader, index):
                raise AssertionError("Disabled video loading must not decode frames")

        namespace = {"torch": torch, "np": np, "os": os, "Path": Path, "random": random,
                     "data": torch.utils.data, "VideoReader": MetadataOnlyVideo,
                     "cpu": lambda value: value}
        definitions = load_definitions("dataset/react_2025.py", {"ReactionDataset", "custom_collate"},
                                       namespace)
        dataset = definitions["ReactionDataset"].__new__(definitions["ReactionDataset"])
        speaker = Path("speaker/session0/clip")
        listener = Path("listener/session0/clip")
        dataset.speaker_path_list = [speaker]
        dataset.listener_path_list = [listener]
        dataset.gt_path_list = [[listener]]
        dataset.pair_manifest = None
        dataset._split = "test"
        dataset._clip_length = 8
        dataset.load_video_s = dataset.load_video_l = False
        dataset.load_audio = dataset.load_emotion_s = dataset.load_emotion_l = True
        dataset.load_3dmm_s = dataset.load_3dmm_l = True
        dataset.audio_feature_type = "wav2vec"
        dataset._video_dir = str(self.output / "video-face-crop")
        dataset._emotion_dir = str(self.output / "facial-attributes")
        dataset._3dmm_dir = str(self.output / "coefficients")
        dataset._audio_dir = str(self.output / "audio-features")
        dataset._transform_3dmm = lambda params: (params,)
        for directory, width in ((dataset._emotion_dir, 25), (dataset._3dmm_dir, 58),
                                 (dataset._audio_dir, 768)):
            for sequence in (speaker, listener):
                path = (Path(directory) / sequence).with_suffix(".npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.ones((8, width), dtype=np.float32))
        item = dataset[0]
        self.assertEqual(len(video_paths), 1)
        self.assertIn("speaker", Path(video_paths[0]).parts)
        self.assertEqual(item[1].numel(), 0)
        self.assertEqual(len(item[4]), 10)
        self.assertTrue(all(video.numel() == 0 for video in item[4]))
        self.assertEqual(len(item[5]), 10)
        batch = definitions["custom_collate"]([item])
        self.assertEqual(batch[1], [])
        self.trainer([batch]).test()
        self.assertEqual(len(self.model_calls), 10)


class MetricInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def metric(*args, **kwargs):
            self.calls.append(args)
            return 0.25

        namespace = {"time": time}
        for name in ("compute_FRC", "compute_FRD", "compute_TLCC", "compute_s_mse", "compute_FRVar"):
            namespace[name] = metric
        self.compute = load_definitions("framework/utils/compute_metrics.py", {"compute_metrics"},
                                        namespace)["compute_metrics"]

    def test_empty_predictions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no predictions were generated"):
            self.compute([], [], [])
        self.assertEqual(self.calls, [])

    def test_mismatched_sample_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample counts must match"):
            self.compute([torch.ones(8, 25)], [torch.ones(10, 8, 25)], [])
        self.assertEqual(self.calls, [])

    def test_no_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no prediction candidates"):
            self.compute([torch.ones(8, 25)], [torch.empty(0, 8, 25)], [torch.ones(10, 8, 25)])
        self.assertEqual(self.calls, [])

    def test_valid_samples_keep_existing_metrics(self):
        report = self.compute([torch.ones(8, 25)], [torch.ones(10, 8, 25)], [torch.ones(10, 8, 25)])
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(report["FRC"], 0.25)
        self.assertEqual(report["smse"], 0.25)


if __name__ == "__main__":
    unittest.main()
