# FlowREACT

FlowREACT generates multiple listener facial reactions from speaker audio and 3DMM motion using windowed flow matching. This source release supports the REACT2025 dataset only and builds on the [REACT2025 baseline](https://github.com/reactmultimodalchallenge/baseline_react2025).

[中文说明](README_zh.md) | [Data preparation](docs/DATA.md) | [Evaluation](docs/EVALUATION.md) | [Release scope](docs/RELEASE_SCOPE.md)

## Contents

- `main.py`, `configs/`, `trainer/motion_flow_fft.py`: Hydra entry point, experiment configuration, and trainer.
- `framework/motion_flow/FlowMatchingVAEV41.py`: FlowREACT model.
- `framework/motion_flow/Emotion3DMMTrans.py`: 3DMM-to-emotion converter.
- `dataset/react_2025.py`, `dataset/data_preprocess/`: REACT2025 loader and audio preprocessing.
- `framework/metrics/`: listener-reaction evaluation metrics.
- `trainer/motion_flow_fft.py`: training, testing, and model profiling.

## Environment

Use Python 3.10 on Linux with a CUDA-capable GPU for training. Install a matching PyTorch, torchvision, and torchaudio build for your CUDA environment, then run:

```bash
python -m pip install -r requirements.txt
```

PyTorch3D and the FaceVerse/PIRender source files are only needed for optional face rendering. They are not bundled here. See [Data preparation](docs/DATA.md).

## Train on REACT2025

Place the REACT2025 dataset, its aligned audio features, and FaceVerse normalization files as described in [Data preparation](docs/DATA.md). Train the 3DMM-to-emotion converter first:

```bash
export DATA_DIR=/path/to/react2025
CUDA_VISIBLE_DEVICES=0 python main.py \
  data=motion_transfer trainer=motion_transfer \
  task=online stage=fit data_dir="$DATA_DIR" \
  seed=1234 run_id=converter_seed1234
```

The converter is written under `save/motion_transfer/react_2025/online/checkpoints/converter_seed1234/Emotion3DMMTransformer/`. Use its selected checkpoint to train FlowREACT:

```bash
export CONVERTER=/path/to/Emotion3DMMTransformer/checkpoint_best.pth
CUDA_VISIBLE_DEVICES=0 python main.py \
  data=motion_flow_fft trainer=motion_flow_fft \
  task=online stage=fit data_dir="$DATA_DIR" \
  seed=1234 run_id=flowreact_seed1234 \
  trainer.model.emotion_transfer_ckpt="$CONVERTER"
```

The default REACT2025 configuration uses eight-frame windows. Model checkpoints and run outputs are written to ignored local directories.

## Evaluate on the validation split

The official REACT2025 test annotations are not provided in this package. With the trained generator and converter available:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  data=motion_flow_fft trainer=motion_flow_fft \
  task=online stage=test data_dir="$DATA_DIR" \
  resume_id=flowreact_seed1234 \
  trainer.model.emotion_transfer_ckpt="$CONVERTER" \
  data.test_dataset.split=val \
  data.test_dataset.load_video_s=false \
  data.test_dataset.load_video_l=false
```

The test path also requires the baseline post-processor checkpoint at `pretrained_models/post_processor/checkpoint.pth`. For evaluation and profiling details, see [Evaluation](docs/EVALUATION.md).

## Notes

Windowed model-forward FPS excludes raw audio feature extraction, disk loading, rendering, and metric computation. Match the window size and preprocessing protocol to the checkpoint used for evaluation. The original dataset, trained weights, and third-party rendering assets are not distributed with this source package.

The original baseline's license notice is retained in [LICENSE](LICENSE). See [Release scope](docs/RELEASE_SCOPE.md) for included code, external assets, and the GAN experiment caveat.
