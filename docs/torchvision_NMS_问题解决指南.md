# 🔧 torchvision::NMS 问题解决指南

## 问题描述

在某些设备上运行时出现以下错误：
```
torchvision::NMS does not exist
```

## 问题原因

1. **torchvision版本过旧** - 较新版本移除了NMS
2. **PyTorch与torchvision版本不匹配** - 版本兼容性问题
3. **编译问题** - 某些平台上的编译问题
4. **CUDA版本不匹配** - GPU环境下的版本冲突

## 🛠️ 解决方案

### 方案1: 安装兼容版本组合

```bash
# 卸载当前版本
pip uninstall torch torchvision

# 安装兼容版本
pip install torch==2.0.1 torchvision==0.15.2
pip install transformers==4.50.0
```

### 方案2: 使用自定义NMS实现

如果无法更改版本，可以使用我们提供的自定义NMS实现：

```python
from src.core.torchvision_compat import safe_nms

# 使用安全的NMS调用
keep_indices = safe_nms(boxes, scores, iou_threshold=0.5)
```

### 方案3: 检查并修复环境

```bash
# 运行兼容性检查
python scripts/check_compatibility.py

# 运行torchvision检查
python src/core/torchvision_compat.py
```

## 📋 推荐的版本组合

| 组件 | 推荐版本 | 说明 |
|------|----------|------|
| PyTorch | 2.0.1 | 稳定版本 |
| torchvision | 0.15.2 | 与PyTorch 2.0.1兼容 |
| transformers | 4.50.0 | 支持qwen2_5_vl模型 |

## 🔍 诊断步骤

### 1. 检查当前版本
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
```

### 2. 检查NMS可用性
```python
import torchvision
if hasattr(torchvision.ops, 'nms'):
    print("✅ NMS可用")
else:
    print("❌ NMS不可用")
```

### 3. 测试NMS功能
```python
import torch
import torchvision.ops

# 创建测试数据
boxes = torch.tensor([[0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5]])
scores = torch.tensor([0.9, 0.8])

# 测试NMS
try:
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print("✅ NMS测试通过")
except Exception as e:
    print(f"❌ NMS测试失败: {e}")
```

## 🚀 快速修复脚本

创建 `fix_torchvision.sh` 脚本：

```bash
#!/bin/bash
echo "修复torchvision NMS问题..."

# 检查当前版本
echo "当前版本:"
python -c "import torch, torchvision; print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}')"

# 安装兼容版本
echo "安装兼容版本..."
pip install torch==2.0.1 torchvision==0.15.2

# 验证修复
echo "验证修复..."
python -c "
import torchvision
if hasattr(torchvision.ops, 'nms'):
    print('✅ NMS问题已修复')
else:
    print('❌ NMS问题仍然存在')
"
```

## ⚠️ 注意事项

1. **版本兼容性**: 确保PyTorch和torchvision版本匹配
2. **CUDA版本**: 如果使用GPU，确保CUDA版本兼容
3. **依赖冲突**: 某些库可能要求特定版本的torchvision
4. **平台差异**: 不同操作系统可能有不同的兼容性问题

## 🔄 备用方案

如果所有方案都失败，可以：

1. **使用Docker**: 使用预配置的Docker镜像
2. **虚拟环境**: 创建独立的Python环境
3. **自定义实现**: 使用我们提供的自定义NMS实现

## 📞 获取帮助

如果问题仍然存在，请：

1. 运行兼容性检查脚本
2. 提供错误信息和环境信息
3. 检查是否使用了推荐的版本组合

## 示例错误信息

```
RuntimeError: torchvision::NMS does not exist
```

这通常表示torchvision版本不兼容，需要安装兼容版本。
