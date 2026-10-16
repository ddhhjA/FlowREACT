# REACT2025 evaluation

Run commands from the repository root. Use the REACT2025 data, normalization files, converter, and FlowREACT checkpoint produced with the same configuration; see [DATA.md](DATA.md). The official test annotations are not included, so the example below evaluates the validation split.

```bash
export DATA_DIR=/path/to/react2025
export CONVERTER=/path/to/Emotion3DMMTransformer/checkpoint_best.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
  data=motion_flow_fft trainer=motion_flow_fft \
  task=online stage=test data_dir="$DATA_DIR" \
  resume_id=flowreact_seed1234 \
  trainer.model.emotion_transfer_ckpt="$CONVERTER" \
  data.test_dataset.split=val \
  data.test_dataset.load_video_s=false \
  data.test_dataset.load_video_l=false
```

The run ID must match the directory containing the trained generator checkpoint. The default generator window is eight frames. The validator writes generated reactions and evaluation metrics to the run output directory. Evaluation requires the baseline post-processor checkpoint described in [DATA.md](DATA.md).

## Additional analyses

`domain_gap_upper_bound.py` evaluates a frozen 3DMM-to-emotion converter on ground-truth listener motion. `analyze_blink_duration.py` measures aggregate eye-closure events. `analyze_expression_groups.py` groups validation inputs by speaker expression. For an isolated-split study, `dataset/data_preprocess/prepare_speaker_split.py` and `prepare_recording_split.py` create audit manifests, and `run_speaker_independent.py` and `run_recording_independent.py` train on those splits. Run each utility with `--help` for its required arguments.

The trainer includes optional model and training profilers. Report any forward-only throughput separately from audio feature extraction, loading, rendering, and metric computation. This source release does not include trained weights or a standalone end-to-end FPS claim.
