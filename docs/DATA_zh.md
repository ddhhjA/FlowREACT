# 数据与 checkpoint

请从 [REACT2025 挑战赛基线项目](https://github.com/reactmultimodalchallenge/baseline_react2025)获取数据，并遵守其数据使用协议。本次源码发布不包含数据集。

## REACT2025

数据加载器要求 `data_dir` 下有 `train/` 和 `val/`。每个划分应采用挑战赛的数据目录结构，例如：

```text
react2025/
  train/
    audio-features/speaker/session*/<clip>.npy
    video-face-crop/speaker/session*/<clip>.mp4
    facial-attributes/speaker/session*/<clip>.npy
    coefficients/speaker/session*/<clip>.npy
    ... 听众对应文件 ...
  val/
    ... 相同的目录结构 ...
```

即使设置 `load_video_s=false` 和 `load_video_l=false`，数据加载器仍通过视频文件名识别对话配对；关闭视频加载只会跳过视频帧解码。缓存的音频特征需要与视频帧对齐，且每帧为 768 维；3DMM 系数为 58 维，面部属性标签为 25 维。可通过以下命令从官方 WAV 文件生成音频特征缓存：

```bash
python dataset/data_preprocess/audio_feature_extraction.py \
  --root_dir /path/to/react2025/train \
  --wav2vec_model_path /path/to/wav2vec2-base-960h
```

验证集也需运行同一命令，并将 `--root_dir` 改为 `/path/to/react2025/val`。脚本会按对应 `video-raw` 文件的帧数对齐输出，因此提取特征时必须备有官方音频和视频文件。

使用零中心化的 3DMM 数据加载器需要 `external/FaceVerse/mean_face.npy` 和 `external/FaceVerse/std_face.npy`。请从官方基线 README 中的链接获取，并放在上述位置。可选的人脸渲染功能还需要另行取得 FaceVerse、PIRender 的源码和权重。在 REACT2025 上运行 `stage=test` 时，还需要基线后处理器文件 `pretrained_models/post_processor/checkpoint.pth`。

请先训练 `Emotion3DMMTransformer`，或提供兼容的预训练转换器。生成模型配置通过 `trainer.model.emotion_transfer_ckpt=/absolute/path/to/checkpoint_best.pth` 指定其 checkpoint。

## 未包含的文件

本源码包不包含原始视频、音频、人脸图像、参与者信息、训练后的 checkpoint、FaceVerse 模型数据、PIRender 资源或中间评估结果。`.gitignore` 也会阻止常见的数据文件和 checkpoint 格式被意外提交。
