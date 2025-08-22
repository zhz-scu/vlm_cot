# .gitignore 文件说明

## 📋 概述

本项目的 `.gitignore` 文件已经配置为排除所有不需要的文件，确保代码仓库保持整洁。

## 🚫 被忽略的文件类型

### 1. **Python相关**
- `__pycache__/` - Python缓存目录
- `*.pyc`, `*.pyo`, `*.pyd` - Python字节码文件
- `build/`, `dist/`, `*.egg-info/` - 构建和分发文件
- `venv/`, `env/`, `.venv/` - 虚拟环境

### 2. **模型文件**
- `models/` - 模型目录
- `*.safetensors`, `*.bin`, `*.pth`, `*.pt`, `*.ckpt` - 模型权重文件
- `scripts/` - 包含下载模型的脚本目录

### 3. **运行时生成的文件**
- `roi_visualization*.png` - ROI可视化图像
- `roi_visualization.png` - ROI可视化图像
- `roi_visualization_demo.png` - 演示ROI可视化
- `cos_vot_visualization.png` - CoS+VoT可视化
- `spatial_visualization.txt` - 空间可视化文本

### 4. **系统文件**
- `.DS_Store` - macOS系统文件
- `Thumbs.db` - Windows缩略图文件
- `*~` - Linux临时文件

### 5. **IDE和编辑器**
- `.idea/` - PyCharm配置
- `.vscode/` - VS Code配置
- `*.swp`, `*.swo` - Vim临时文件

### 6. **临时和缓存文件**
- `temp/`, `tmp/`, `*.tmp` - 临时文件
- `.cache/`, `cache/`, `*.cache` - 缓存文件
- `*.log` - 日志文件
- `output/`, `results/` - 输出文件

### 7. **配置和敏感文件**
- `.env`, `.env.local` - 环境变量文件
- `config.ini`, `secrets.json` - 配置文件
- `*.zip`, `*.tar.gz` - 压缩文件

## ✅ 保留的文件

### 1. **示例图像**
- `test_simple.png` - 测试用简单图像
- `demo_complex_scene.png` - 演示用复杂场景图像

### 2. **文档图像**
- `docs/images/*.png` - 文档中的图像
- `docs/images/*.jpg` - 文档中的图像

## 🔧 如何添加新文件到忽略列表

### 1. 添加新的忽略规则
在 `.gitignore` 文件中添加新的规则：

```bash
# 新文件类型
*.new_extension
new_directory/
```

### 2. 忽略已跟踪的文件
如果文件已经被Git跟踪，需要先取消跟踪：

```bash
git rm --cached filename
git rm -r --cached directory/
```

### 3. 强制添加被忽略的文件
如果需要添加被忽略的文件：

```bash
git add -f filename
```

## 📊 当前忽略状态

运行以下命令查看当前被忽略的文件：

```bash
git status --ignored
```

## 🎯 最佳实践

### 1. **定期检查**
- 定期运行 `git status --ignored` 检查忽略状态
- 确保没有意外忽略重要文件

### 2. **项目特定规则**
- 在项目特定部分添加相关规则
- 使用注释说明规则的目的

### 3. **测试忽略规则**
- 在添加新规则前测试其效果
- 确保规则不会意外忽略重要文件

### 4. **文档化**
- 在README中说明重要的忽略规则
- 为团队成员提供清晰的指导

## 🚨 注意事项

### 1. **模型文件**
- 模型文件通常很大，不应该提交到Git
- 使用模型下载脚本或外部存储

### 2. **敏感信息**
- 确保包含API密钥、密码等敏感信息的文件被忽略
- 使用环境变量或配置文件模板

### 3. **临时文件**
- 运行时生成的临时文件应该被忽略
- 但示例和测试文件应该保留

### 4. **平台特定文件**
- 不同操作系统的系统文件应该被忽略
- 确保跨平台兼容性

## 📝 更新日志

- **v1.0**: 初始版本，包含基本的Python和项目特定规则
- **v1.1**: 添加了运行时生成文件的忽略规则
- **v1.2**: 优化了模型文件和缓存文件的忽略规则

---

**通过合理的 `.gitignore` 配置，我们可以保持代码仓库的整洁，避免提交不必要的文件！** 🚀
