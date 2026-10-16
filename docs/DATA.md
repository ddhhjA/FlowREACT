# Data and checkpoints

Download REACT2025 from the [challenge baseline page](https://github.com/reactmultimodalchallenge/baseline_react2025) and follow its data-use agreement. The dataset is not included in this source release.

## REACT2025

The data loader expects `train/` and `val/` under `data_dir`. Each split should contain the challenge layout:

```text
react2025/
  train/
    audio-features/speaker/session*/<clip>.npy
    video-face-crop/speaker/session*/<clip>.mp4
    facial-attributes/speaker/session*/<clip>.npy
    coefficients/speaker/session*/<clip>.npy
    ... corresponding listener files ...
  val/
    ... same directory structure ...
```

The loader uses video filenames to discover conversation pairs even when `load_video_s=false` and `load_video_l=false`; disabling video loading only avoids frame decoding. The audio cache must be frame-aligned and have 768 features per frame; 3DMM coefficients use 58 dimensions and the facial-attribute labels use 25 dimensions. To create the audio cache from the official WAV files, run:

```bash
python dataset/data_preprocess/audio_feature_extraction.py \
  --root_dir /path/to/react2025/train \
  --wav2vec_model_path /path/to/wav2vec2-base-960h
```

Run the same command with `--root_dir /path/to/react2025/val` for validation. This script aligns the output to the corresponding `video-raw` frame count, so the official audio and video files must be present during extraction.

The zero-centered 3DMM loader requires `external/FaceVerse/mean_face.npy` and `external/FaceVerse/std_face.npy`. Obtain them from the links in the official baseline README and place them at those exact paths. The optional renderer needs separately obtained FaceVerse and PIRender source and weights. `stage=test` on REACT2025 also requires the baseline post-processor at `pretrained_models/post_processor/checkpoint.pth`.

Train `Emotion3DMMTransformer` before FlowREACT, or provide a compatible pretrained converter. The generator configuration requires its checkpoint via `trainer.model.emotion_transfer_ckpt=/absolute/path/to/checkpoint_best.pth`.

## Files intentionally absent

No source videos, audio files, face images, participant information, trained checkpoints, FaceVerse model data, PIRender assets, or intermediate evaluation results are packaged. The `.gitignore` also prevents accidental commits of common data and checkpoint formats.
