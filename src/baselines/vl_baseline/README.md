# 原始VL模型基线 (无CoT/MCOT)

本目录提供不使用思维链/多步推理的“原始VL模型”推理脚本，便于与CoT/MCOT方法做对比评测。

## 文件结构
- `infer_vl_baseline.py`: 原始VL推理脚本（支持CPU/CUDA/MPS/NPU，支持本地模型目录）

## 依赖
- PyTorch >= 2.2
- Transformers 4.50.x
- qwen-vl-utils >= 0.0.8

## 调用示例

### 1) 使用HuggingFace模型ID
```bash
python src/baselines/vl_baseline/infer_vl_baseline.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --image test_simple.png \
  --question "图片中有什么？" \
  --device auto --dtype auto --max-new-tokens 256
```

### 2) 使用本地已下载模型
```bash
python src/baselines/vl_baseline/infer_vl_baseline.py \
  --model-id models/Qwen2.5-VL-3B-Instruct/qwen/Qwen2.5-VL-3B-Instruct \
  --image test_simple.png \
  --question "图片中有什么？" \
  --device mps --dtype fp16
```

### 3) 指定不同设备
- `--device cpu|cuda|mps|npu|xpu|auto`
- `--dtype auto|fp16|bf16|fp32`

## 输出示例
```
=== VL Baseline 结果 ===
设备: mps
精度: torch.float16
答案: 一个蓝色方形在红色背景上。
```

## 与CoT/MCOT对比建议
- 同一张图像与问题，分别运行：
  - `src/baselines/vl_baseline/infer_vl_baseline.py`
  - `src/basic_cot/infer_cot.py` 或 `src/chain_of_spot/cos_vot_npu.py`
- 对比：
  - 答案质量与细节
  - 生成时延与显存
  - ROI/可视化轨迹（若适用）
