# FlowREACT

FlowREACT 使用基于窗口的流匹配方法，根据说话人的音频和 3DMM 运动生成多种听众面部反应。本次源码发布仅支持 REACT2025 数据集，代码基于 [REACT2025 基线项目](https://github.com/reactmultimodalchallenge/baseline_react2025)。

[English](README.md) | [数据准备](docs/DATA_zh.md) | [评估说明](docs/EVALUATION_zh.md) | [发布范围](docs/RELEASE_SCOPE_zh.md)

## 代码结构

- `main.py`、`configs/`、`trainer/motion_flow_fft.py`：Hydra 入口、实验配置及训练器。
- `framework/motion_flow/FlowMatchingVAEV41.py`：FlowREACT 模型。
- `framework/motion_flow/Emotion3DMMTrans.py`：3DMM 到情绪标签的转换器。
- `dataset/react_2025.py`、`dataset/data_preprocess/`：REACT2025 数据加载与音频预处理。
- `framework/metrics/`：听众反应评估指标。
- `trainer/motion_flow_fft.py`：训练、测试及模型性能分析。

## 运行环境

训练建议使用 Linux、Python 3.10 和支持 CUDA 的 GPU。先安装与本机 CUDA 环境匹配的 PyTorch、torchvision 和 torchaudio，再运行：

```bash
python -m pip install -r requirements.txt
```

PyTorch3D 以及 FaceVerse/PIRender 源码仅用于可选的人脸渲染功能，本仓库不提供这些文件。详见[数据准备](docs/DATA_zh.md)。

## 在 REACT2025 上训练

按[数据准备](docs/DATA_zh.md)放置 REACT2025 数据集、对齐的音频特征及 FaceVerse 归一化文件。首先训练 3DMM 到情绪标签的转换器：

```bash
export DATA_DIR=/path/to/react2025
CUDA_VISIBLE_DEVICES=0 python main.py \
  data=motion_transfer trainer=motion_transfer \
  task=online stage=fit data_dir="$DATA_DIR" \
  seed=1234 run_id=converter_seed1234
```

转换器保存在 `save/motion_transfer/react_2025/online/checkpoints/converter_seed1234/Emotion3DMMTransformer/` 下。选择其 checkpoint 后训练 FlowREACT：

```bash
export CONVERTER=/path/to/Emotion3DMMTransformer/checkpoint_best.pth
CUDA_VISIBLE_DEVICES=0 python main.py \
  data=motion_flow_fft trainer=motion_flow_fft \
  task=online stage=fit data_dir="$DATA_DIR" \
  seed=1234 run_id=flowreact_seed1234 \
  trainer.model.emotion_transfer_ckpt="$CONVERTER"
```

REACT2025 的默认配置采用 8 帧窗口。模型 checkpoint 和运行输出保存在本地目录中，这些目录已列入 `.gitignore`。

## 在验证集上评估

本源码包不包含 REACT2025 官方测试集标注。准备好训练后的生成模型和转换器后，运行：

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

测试还需要基线后处理器的 checkpoint：`pretrained_models/post_processor/checkpoint.pth`。评估和性能分析的其他说明见[评估说明](docs/EVALUATION_zh.md)。

## 说明

按窗口统计的模型前向 FPS 不包含原始音频特征提取、磁盘读取、渲染或指标计算。评估时的窗口大小和预处理协议应与对应 checkpoint 一致。本源码包不分发原始数据集、训练权重或第三方渲染资源。

仓库保留了原始基线的 [LICENSE](LICENSE) 声明。所含代码、外部资源及 GAN 实验代码的说明见[发布范围](docs/RELEASE_SCOPE_zh.md)。
