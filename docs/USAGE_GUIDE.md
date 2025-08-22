# VLM CoT 使用指南

## 📋 目录
1. [快速开始](#快速开始)
2. [核心功能](#核心功能)
3. [参数详解](#参数详解)
4. [使用示例](#使用示例)
5. [性能优化](#性能优化)
6. [常见问题](#常见问题)

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 下载模型（可选，首次运行会自动下载）
python scripts/download_model.py
```

### 2. 基础使用
```bash
# 最简单的使用方式
python src/infer_cot.py \
  --image your_image.jpg \
  --question "描述这张图片"
```

## 🎯 核心功能

### 支持的 CoT 风格
1. **`rationale_and_answer`** - 显式推理过程 + 最终答案
2. **`short_answer`** - 隐藏推理，仅输出答案
3. **`free`** - 自由格式，不加约束

### 设备支持
- **CUDA** - NVIDIA GPU 加速
- **MPS** - Apple Silicon Mac 加速（推荐）
- **CPU** - 通用计算设备

## ⚙️ 参数详解

### 必需参数
| 参数 | 说明 | 示例 |
|------|------|------|
| `--image` | 图片路径或URL | `--image photo.jpg` |

### 常用参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--question` | "请逐步思考并回答问题" | 要询问的问题 |
| `--cot-style` | `rationale_and_answer` | CoT输出风格 |
| `--device` | `auto` | 计算设备选择 |
| `--dtype` | `auto` | 计算精度 |
| `--max-new-tokens` | `512` | 最大生成token数 |
| `--temperature` | `0.2` | 采样温度 |
| `--top-p` | `0.9` | 核采样阈值 |

### 高级参数
| 参数 | 说明 |
|------|------|
| `--model-id` | 模型标识（默认：Qwen/Qwen2.5-VL-3B-Instruct） |
| `--seed` | 随机种子 |
| `--json` | JSON格式输出 |

## 📝 使用示例

### 示例1：基础图片描述
```bash
python src/infer_cot.py \
  --image cat.jpg \
  --question "这只猫在做什么？" \
  --cot-style rationale_and_answer
```

### 示例2：多图片对比
```bash
python src/infer_cot.py \
  --image image1.jpg --image image2.jpg \
  --question "两张图片有什么不同？" \
  --cot-style short_answer
```

### 示例3：Mac MPS 优化
```bash
python src/infer_cot.py \
  --image document.jpg \
  --question "提取图片中的文字信息" \
  --device mps --dtype fp16 \
  --max-new-tokens 256
```

### 示例4：科学问题推理
```bash
python src/scienceqa_cot.py \
  --image science_diagram.jpg \
  --question "解释这个物理现象" \
  --context "这是一个电路图"
```

### 示例5：高级推理功能
```bash
python src/advanced_cot.py \
  --image complex_image.jpg \
  --question "分析图片中的逻辑关系" \
  --enable-advanced-features
```

## ⚡ 性能优化

### Mac MPS 优化
```bash
# 推荐配置
python src/infer_cot.py \
  --image your_image.jpg \
  --question "your question" \
  --device mps --dtype fp16 \
  --max-new-tokens 256
```

### 内存优化
```bash
# 降低内存使用
python src/infer_cot.py \
  --image your_image.jpg \
  --question "your question" \
  --max-new-tokens 128 \
  --device cpu
```

### 批量处理
```bash
# 使用脚本进行批量测试
python scripts/vlm_cot_test.py
```

## 🔧 常见问题

### 设备相关问题

**Q: MPS 不可用怎么办？**
```bash
# 检查 MPS 状态
python scripts/test_mps.py

# 强制使用 CPU
python src/infer_cot.py --device cpu
```

**Q: 显存不足怎么办？**
```bash
# 降低参数
python src/infer_cot.py \
  --max-new-tokens 64 \
  --device cpu
```

### 模型相关问题

**Q: 模型下载慢怎么办？**
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
python src/infer_cot.py ...

# 或使用 ModelScope
python scripts/download_model.py
```

**Q: 模型加载失败怎么办？**
```bash
# 升级 transformers
pip install --upgrade transformers

# 或从源码安装
pip install git+https://github.com/huggingface/transformers
```

### 图片相关问题

**Q: 网络图片无法访问？**
```bash
# 下载到本地
curl -o local_image.jpg "https://example.com/image.jpg"
python src/infer_cot.py --image local_image.jpg
```

**Q: 图片格式不支持？**
```bash
# 转换为支持的格式
pip install Pillow
python -c "
from PIL import Image
img = Image.open('your_image.png')
img.save('converted.jpg', 'JPEG')
"
```

## 🧪 测试和调试

### 运行测试
```bash
# 基础功能测试
python scripts/test_mps.py

# CoT 对比测试
python scripts/vlm_cot_test.py

# 生成测试报告
python scripts/cot_final_report.py
```

### 调试技巧
```bash
# 查看详细日志
python src/infer_cot.py --image test.jpg --question "test" 2>&1 | tee debug.log

# 检查设备状态
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
"
```

## 📊 性能基准

### 设备性能对比
| 设备 | 推理速度 | 内存使用 | 推荐场景 |
|------|----------|----------|----------|
| MPS (Mac) | 快 | 中等 | 日常使用 |
| CUDA (NVIDIA) | 最快 | 高 | 批量处理 |
| CPU | 慢 | 低 | 调试/测试 |

### 参数影响
| 参数 | 速度影响 | 质量影响 | 建议 |
|------|----------|----------|------|
| `max-new-tokens` | 线性 | 显著 | 根据问题复杂度调整 |
| `temperature` | 无 | 显著 | 0.1-0.3 推荐 |
| `device` | 显著 | 无 | 优先 MPS/CUDA |

## 🔗 相关资源

- [技术白皮书](TECHNICAL_WHITEPAPER.md)
- [项目 README](../README.md)
- [Qwen2.5-VL 官方文档](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [ScienceQA 论文](https://arxiv.org/abs/2203.10227)
