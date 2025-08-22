# Chain-of-Spot 使用指南

## 🎯 什么是 Chain-of-Spot？

Chain-of-Spot (CoS) 是一种创新的**交互式推理方法**，通过两步推理过程显著提升视觉语言模型的理解能力：

1. **步骤1**: 自动识别图像中与问题相关的关键区域 (ROI)
2. **步骤2**: 基于ROI和原图生成更精确、详细的答案

## 🚀 快速开始

### 1. 基础使用

```bash
# 基本命令
python src/chain_of_spot/cos_inference.py \
  --image your_image.jpg \
  --question "请描述图像中的主要对象" \
  --device mps --dtype fp16

# 保存ROI可视化
python src/chain_of_spot/cos_inference.py \
  --image your_image.jpg \
  --question "请描述绿色圆形的位置和特征" \
  --save-roi-viz \
  --output-dir ./results
```

### 2. Python API 使用

```python
from src.chain_of_spot import cos_generate
from PIL import Image

# 简单使用
result = cos_generate(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    image_path="image.jpg",
    question="描述图像中的主要特征",
    device="mps"
)

print(f"ROI区域: {result['roi_bbox']}")
print(f"答案: {result['final_answer']}")
```

## 📊 与其他方法对比

| 特性 | 基础CoT | Chain-of-Spot |
|------|---------|---------------|
| **ROI聚焦** | ❌ 全局分析 | ✅ 动态识别 |
| **细节水平** | 中等 | 高 |
| **位置信息** | 模糊 | 精确 |
| **推理过程** | 单步 | 交互式两步 |
| **适用场景** | 一般描述 | 细节分析 |

## 🎨 演示效果

运行演示脚本查看效果对比：

```bash
python demos/chain_of_spot_demo.py
```

**演示输出**:
- 创建包含多个几何图形的测试图像
- 对比基础CoT和Chain-of-Spot的推理效果
- 生成ROI可视化图像

## 🔧 参数说明

### 命令行参数

- `--image`: 输入图像路径 (必需)
- `--question`: 要询问的问题 (必需)
- `--device`: 推理设备 (`auto`, `cuda`, `mps`, `cpu`)
- `--dtype`: 数据类型 (`auto`, `bf16`, `fp16`, `fp32`)
- `--max-new-tokens`: 最大生成token数 (默认: 512)
- `--save-roi-viz`: 保存ROI可视化
- `--output-dir`: 输出目录
- `--json`: JSON格式输出

### API 参数

```python
cos_generate(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",  # 模型ID
    image_path="image.jpg",                   # 图像路径
    question="描述主要对象",                    # 问题
    device="auto",                            # 设备
    dtype_str="auto",                         # 数据类型
    max_new_tokens=512,                       # 最大token数
    seed=None,                                # 随机种子
    save_roi_visualization=False,             # 保存可视化
    output_dir="."                            # 输出目录
)
```

## 🎯 最佳实践

### 1. 问题类型
Chain-of-Spot 特别适合以下类型的问题：

✅ **推荐**:
- "描述图像中红色汽车的特征"
- "分析左下角的文字内容"
- "这个绿色物体的具体位置在哪里？"

❌ **不推荐**:
- "图像整体的主题是什么？"
- "总共有多少个物体？"

### 2. 设备选择
- **Mac**: 使用 `--device mps --dtype fp16`
- **NVIDIA GPU**: 使用 `--device cuda --dtype bf16`
- **CPU**: 使用 `--device cpu --dtype fp32`

### 3. 性能优化
```bash
# 快速推理 (降低质量)
--max-new-tokens 128 --dtype fp16

# 高质量推理 (较慢)
--max-new-tokens 512 --dtype fp32
```

## 📈 技术优势

### 🔍 动态ROI识别
- 自动聚焦问题相关区域
- 避免无关信息干扰
- 提供精确的位置信息

### 🎨 多粒度特征
- 保持原图分辨率
- 获取局部细节信息
- 无需增加计算成本

### ⚡ 交互式推理
- 两步推理过程
- 可解释的推理轨迹
- 更准确的答案生成

## 🔬 高级用法

### 1. 批量处理
```python
from src.chain_of_spot import ChainOfSpotModel
from PIL import Image

# 加载模型
model, processor = load_model_and_processor(...)
cos_model = ChainOfSpotModel(model, processor)

# 批量推理
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
questions = ["描述主要对象"] * 10
results = cos_model.batch_reasoning(images, questions)
```

### 2. 自定义ROI处理
```python
from src.chain_of_spot import BoundingBox, ImageCropper

# 手动指定ROI
roi_bbox = BoundingBox(x0=0.2, x1=0.8, y0=0.1, y1=0.7)
cropped_image = ImageCropper.crop_image(image, roi_bbox)

# 可视化ROI
viz_image = ImageCropper.visualize_roi(image, roi_bbox, color="red")
```

### 3. 结果分析
```python
# 分析推理轨迹
for i, trace in enumerate(result['reasoning_trace']):
    print(f"步骤 {i+1}: {trace}")

# 可视化置信度
print(f"推理置信度: {result['confidence']:.3f}")
```

## 🛠️ 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查模型是否存在
   python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')"
   ```

2. **设备不支持**
   ```bash
   # 检查MPS支持
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

3. **内存不足**
   ```bash
   # 使用CPU推理
   --device cpu --max-new-tokens 128
   ```

4. **ROI识别失败**
   - 检查问题是否明确指向特定区域
   - 尝试更具体的问题描述

### 调试技巧

```bash
# 启用详细日志
python src/chain_of_spot/cos_inference.py \
  --image image.jpg \
  --question "问题" \
  --json 2>&1 | tee debug.log
```

## 📚 更多资源

- **技术文档**: [CHAIN_OF_SPOT_TECHNICAL.md](CHAIN_OF_SPOT_TECHNICAL.md)
- **演示脚本**: `demos/chain_of_spot_demo.py`
- **源代码**: `src/chain_of_spot/`
- **论文**: "Chain-of-Spot: Interactive Reasoning Improves Large Vision-Language Models"

---

**开始使用 Chain-of-Spot，体验交互式推理的强大能力！** 🚀
