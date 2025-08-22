# VLM CoT with Qwen2.5-VL-3B-Instruct

基于 Qwen/Qwen2.5-VL-3B-Instruct 的视觉语言模型（VLM）链式思维（Chain-of-Thought, CoT）推理脚本。

**✨ Mac MPS 优化版本** - 针对 Apple Silicon Mac 进行了特别优化，支持 Metal Performance Shaders (MPS) 加速。

## 安装

1. Python >= 3.9
2. 安装依赖：

```bash
pip install -r requirements.txt
```

可选：安装 `flash-attn` 以加速（按需，GPU环境）：

```bash
pip install flash-attn --no-build-isolation
```

## 运行

脚本路径：`src/infer_cot.py`

- 必需参数：`--image`（可多次传入），支持本地路径或URL。
- 常用参数：`--question`、`--cot-style`、`--max-new-tokens`、`--temperature`、`--top-p`、`--device`、`--dtype`。

### Mac MPS 使用示例

**推荐配置（Mac MPS 加速）：**

```bash
python src/infer_cot.py \
  --image https://raw.githubusercontent.com/QwenLM/Qwen2-VL/master/examples/images/ocr_1.jpg \
  --question "请逐步识别图片内容并总结要点" \
  --cot-style rationale_and_answer \
  --device mps --dtype fp16 \
  --max-new-tokens 512 --temperature 0.2 --top-p 0.9
```

**自动设备选择（推荐）：**

```bash
python src/infer_cot.py \
  --image image1.jpg --image image2.jpg \
  --question "两张图片的共同点是什么？" \
  --cot-style short_answer \
  --device auto --dtype auto
```

### 其他示例

示例1：单图CoT（显式"思考过程+最终答案"）

```bash
python src/infer_cot.py \
  --image https://raw.githubusercontent.com/QwenLM/Qwen2-VL/master/examples/images/ocr_1.jpg \
  --question "请逐步识别图片内容并总结要点" \
  --cot-style rationale_and_answer \
  --max-new-tokens 512 --temperature 0.2 --top-p 0.9
```

示例2：多图推理（隐藏中间推理，仅输出最终答案）

```bash
python src/infer_cot.py \
  --image image1.jpg --image image2.jpg \
  --question "两张图片的共同点是什么？" \
  --cot-style short_answer
```

示例3：以JSON输出

```bash
python src/infer_cot.py \
  --image some.jpg \
  --question "图中时间与地点推断？" \
  --cot-style rationale_and_answer \
  --json
```

## Mac MPS 优化说明

### 设备选择策略
- `--device auto`：自动选择最佳设备
  - Mac with Apple Silicon → MPS
  - NVIDIA GPU → CUDA  
  - 其他 → CPU
- `--device mps`：强制使用 MPS（Mac 推荐）
- `--device cpu`：强制使用 CPU（调试用）

### 数据类型优化
- `--dtype auto`：自动选择最佳精度
  - MPS → float16（推荐）
  - CUDA → bfloat16
  - CPU → float32
- `--dtype fp16`：强制使用 float16（MPS 最佳性能）

### 性能建议
1. **首次运行**：使用 `--device cpu` 验证功能
2. **正常使用**：使用 `--device mps --dtype fp16` 获得最佳性能
3. **内存不足**：降低 `--max-new-tokens` 或使用 `--device cpu`

## 文档
- [📖 详细使用指南](docs/USAGE_GUIDE.md) - 完整的使用说明和示例
- [🔬 技术白皮书](docs/TECHNICAL_WHITEPAPER.md) - 技术架构和创新点

## 备注
- `--device auto` 会自动选择 `cuda` > `mps` > `cpu`。
- `--dtype auto`：CUDA默认 `bf16`，MPS默认 `fp16`，CPU默认 `fp32`。
- 模型默认：`Qwen/Qwen2.5-VL-3B-Instruct`，如需本地权重，传入 `--model-id /path/to/model`。

## 数据与格式
脚本内部使用 `processor.apply_chat_template` 和 `qwen_vl_utils.process_vision_info` 处理图像与多轮消息，消息格式：

```json
[
  {
    "role": "user",
    "content": [
      {"type": "image", "image": "<path-or-url>"},
      {"type": "text",  "text":  "<prompt>"}
    ]
  }
]
```

`--cot-style` 取值：
- `rationale_and_answer`: 输出"思考过程：…\n最终答案：…"。
- `short_answer`: 隐藏中间推理，仅输出一句话答案。
- `free`: 使用你提供的 `--question` 原文。

## 常见问题

### Mac 相关问题
- **MPS 不可用**：脚本会自动回退到 CPU，检查 macOS 版本是否支持 MPS
- **首次运行慢**：MPS 需要预热，后续运行会更快
- **内存不足**：尝试降低 `--max-new-tokens` 或使用 `--device cpu`

### 通用问题
- 显存不足：尝试降低 `--max-new-tokens`、`--temperature`，或改用 `--device cpu` 以做功能验证。
- 网络图片403：请改用可公开访问的URL或下载为本地文件路径。
- 模型下载慢：使用 `HF_ENDPOINT=https://hf-mirror.com` 环境变量加速下载。

### 调试技巧
```bash
# 查看详细日志
python src/infer_cot.py --image test.jpg --question "test" --device cpu 2>&1 | tee debug.log

# 检查设备状态
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```
