import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
except ImportError:
    compose = None

from run_speaker_independent import hydra_value, run


@unittest.skipIf(compose is None, "Config composition requires hydra-core")
class SpeakerConfigTests(unittest.TestCase):
    def config(self, trainer, stage="fit", group="unseen", data_name="motion_flow_fft_speaker"):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver("get_last_checkpoint", lambda path: str(root / path / "checkpoints"), replace=True)
        OmegaConf.register_new_resolver("generate_id", lambda: "synthetic", replace=True)
        OmegaConf.register_new_resolver("code_path", lambda path="": str(root / path), replace=True)
        OmegaConf.register_new_resolver("working_path", lambda path: str(root / path), replace=True)
        overrides = ["trainer=" + trainer, "data=" + data_name, "task=online",
                     "stage=" + stage, "data_dir=" + hydra_value("data with spaces"),
                     "data.split_dir=" + hydra_value("split with spaces"),
                     "data.evaluation_group=" + group, "trainer.max_seq_len=256", "trainer.window_size=8"]
        with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
            cfg = compose(config_name="main", overrides=overrides)
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        return cfg

    def test_flow_config_composes_and_preserves_the_architecture(self):
        cfg = self.config("motion_flow_fft_speaker")
        self.assertEqual(cfg.trainer.model._target_, "framework.motion_flow.FlowMatchingVAEV41.FlowReact")
        self.assertEqual(cfg.trainer.model.seq_len, 256)
        self.assertEqual(cfg.trainer.validation_window_size, 8)
        self.assertEqual(cfg.trainer.speaker_independent.training_seed, 1234)
        self.assertEqual(cfg.data.train_dataset.pair_manifest, "split with spaces/train.csv")
        self.assertFalse(cfg.trainer.renderer.do_render)

    def test_seen_and_unseen_load_the_matching_test_manifest(self):
        for group in ("seen", "unseen"):
            cfg = self.config("motion_flow_fft_speaker", "test", group)
            self.assertEqual(cfg.data.test_dataset.pair_manifest, "split with spaces/" + group + ".csv")
            self.assertEqual(cfg.data.test_dataset.split, "test")
            self.assertFalse(cfg.data.test_dataset.shuffle)
            self.assertFalse(cfg.data.test_dataset.load_video_s)
            self.assertFalse(cfg.data.test_dataset.load_video_l)
            self.assertEqual(cfg.trainer.speaker_independent.evaluation_group, group)

    def test_auxiliary_converter_uses_the_same_isolated_datasets(self):
        cfg = self.config("motion_transfer")
        self.assertEqual(cfg.trainer.model._target_, "framework.motion_flow.Emotion3DMMTrans.Emotion3DMMTransformer")
        self.assertEqual(cfg.data.train_dataset.pair_manifest, "split with spaces/train.csv")
        self.assertEqual(cfg.data.validation_dataset.pair_manifest, "split with spaces/val.csv")

    def test_recording_configs_compose_without_identity_claims(self):
        cfg = self.config("motion_flow_fft_recording", "test", "heldout", "motion_flow_fft_recording")
        self.assertEqual(cfg.data.data_name, "react_2025_recording_disjoint")
        self.assertEqual(cfg.trainer.speaker_independent.protocol, "recording_disjoint")
        self.assertEqual(cfg.data.test_dataset.pair_manifest, "split with spaces/heldout.csv")
        self.assertEqual(cfg.trainer.model._target_, "framework.motion_flow.FlowMatchingVAEV41.FlowReact")
        self.assertEqual(cfg.trainer.validation_window_size, 8)

    def test_recording_runner_all_commands_compose(self):
        from test_recording_independent import RecordingIndependentTests
        fixture = RecordingIndependentTests()
        fixture.setUp()
        try:
            split_dir, _ = fixture.prepare()
            commands = []
            with patch("run_speaker_independent.execute", side_effect=lambda command, *a: commands.append(command)):
                run(fixture.args(split_dir))
            self.config("motion_flow_fft_recording", "test", "heldout", "motion_flow_fft_recording")
            root = Path(__file__).resolve().parents[1]
            self.assertEqual(len(commands), 3)
            for command in commands:
                with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
                    cfg = compose(config_name="main", overrides=command[2:])
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                self.assertEqual(cfg.data.data_name, "react_2025_recording_disjoint")
                self.assertEqual(cfg.trainer.validation_window_size, 8)
                self.assertFalse(cfg.resume)
        finally:
            fixture.tearDown()

    def test_complete_runner_commands_compose_without_unknown_overrides(self):
        from test_speaker_independent import SpeakerIndependentTests
        fixture = SpeakerIndependentTests()
        fixture.setUp()
        try:
            split_dir, _ = fixture.prepare()
            args = Namespace(output_dir=str(fixture.root / "outputs"), seeds=[1234], stage="all",
                             data_dir=str(fixture.data), split_dir=str(split_dir), window_size=8,
                             clip_length=256, epochs=1, converter_epochs=1, batch_size=2,
                             workers=0, metric_threads=1, history_weight=1.0,
                             history_mode="previous_window", converter_checkpoint=None, dry_run=True)
            commands = []
            with patch("run_speaker_independent.execute", side_effect=lambda command, *args: commands.append(command)):
                run(args)
            self.assertEqual(len(commands), 4)
            self.assertFalse((fixture.root / "outputs").exists())
            self.config("motion_flow_fft_speaker")  # Register the main.py resolvers.
            root = Path(__file__).resolve().parents[1]
            for command in commands:
                with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
                    cfg = compose(config_name="main", overrides=command[2:])
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                self.assertEqual(cfg.trainer.validation_window_size, 8)
                self.assertEqual(cfg.data.train_dataset.pair_manifest, split_dir.as_posix() + "/train.csv")
                self.assertFalse(cfg.resume)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
