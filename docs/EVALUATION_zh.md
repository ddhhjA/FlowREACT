# REACT2025 评估

请在仓库根目录运行命令。数据、归一化文件、转换器和 FlowREACT checkpoint 应与同一实验配置对应，详见[数据说明](DATA_zh.md)。本源码包不包含官方测试集标注，因此以下示例使用验证集。

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

`resume_id` 必须与训练后生成模型 checkpoint 所在的运行目录一致。生成模型默认使用 8 帧窗口。评估程序将生成的反应和指标写入本次运行的输出目录。评估还需要[数据说明](DATA_zh.md)中列出的基线后处理器 checkpoint。

## 其他分析

`domain_gap_upper_bound.py` 使用真实听众运动评估冻结的 3DMM 到情绪标签转换器。`analyze_blink_duration.py` 统计整体眼部闭合事件；`analyze_expression_groups.py` 按说话者表情对验证集输入分组。对于独立划分实验，`dataset/data_preprocess/prepare_speaker_split.py` 和 `prepare_recording_split.py` 用于生成可核查的划分清单，`run_speaker_independent.py` 和 `run_recording_independent.py` 用于在相应划分上训练。各工具所需的参数和输出选项可通过 `--help` 查看。

训练器还提供可选的模型及训练性能分析。仅包含模型前向计算的吞吐量，应与音频特征提取、数据加载、渲染及指标计算分开报告。本源码包不包含训练权重，也不提供单独的端到端 FPS 结论。
