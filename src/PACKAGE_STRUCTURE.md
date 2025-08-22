# VLM CoT 推理系统 - 包结构说明

## 📦 分包完成情况

✅ **已完成的分包整理**：

### 1. core/ - 核心组件
- ✅ `qwen_vl_utils.py` - Qwen VL模型工具函数
- ✅ `__init__.py` - 包初始化文件

### 2. basic_cot/ - 基础CoT实现
- ✅ `infer_cot.py` - 基础多模态CoT推理
- ✅ `__init__.py` - 包初始化文件

### 3. advanced_cot/ - 高级CoT实现
- ✅ `advanced_cot.py` - 高级多模态推理引擎
- ✅ `advanced_mcot.py` - 多模态CoT实现
- ✅ `enhanced_cot.py` - 增强型CoT
- ✅ `__init__.py` - 包初始化文件

### 4. specialized_cot/ - 专门化CoT实现
- ✅ `scienceqa_cot.py` - ScienceQA科学推理
- ✅ `scienceqa_inspired.py` - ScienceQA启发式推理
- ✅ `hierarchical_cot.py` - 分层CoT推理
- ✅ `__init__.py` - 包初始化文件

### 5. experimental/ - 实验性实现
- ✅ `vot_prompting.py` - Visualization-of-Thought
- ✅ `__init__.py` - 包初始化文件

## 🎯 各包功能特点

### core/ - 核心组件
**功能**: 提供系统的基础工具和组件
- **qwen_vl_utils.py**: 视觉信息处理、模型加载等核心功能
- **特点**: 被其他所有包依赖的基础组件

### basic_cot/ - 基础CoT实现
**功能**: 提供基础的链式思维推理
- **infer_cot.py**: 支持多种输出风格的基础CoT
- **特点**: 针对Mac MPS优化，适合日常使用

### advanced_cot/ - 高级CoT实现
**功能**: 集成多种高新技术的推理引擎
- **advanced_cot.py**: 跨模态注意力机制、置信度校准
- **advanced_mcot.py**: 多步推理、动态上下文累积
- **enhanced_cot.py**: ScienceQA优化、Few-shot学习
- **特点**: 适合复杂推理任务

### specialized_cot/ - 专门化CoT实现
**功能**: 针对特定任务优化的推理方法
- **scienceqa_cot.py**: 科学问题推理，QCM→ALE架构
- **scienceqa_inspired.py**: ScienceQA启发式优化
- **hierarchical_cot.py**: 分层推理、递归结构
- **特点**: 针对特定领域优化

### experimental/ - 实验性实现
**功能**: 实验性的推理方法
- **vot_prompting.py**: Visualization-of-Thought空间推理
- **特点**: 前沿研究，可能不稳定

## 🚀 使用方式

### 1. 直接导入包
```python
from src import basic_cot, advanced_cot, specialized_cot, experimental, core
```

### 2. 导入具体功能
```python
# 基础CoT
from src.basic_cot.infer_cot import generate

# 高级CoT
from src.advanced_cot.advanced_cot import advanced_generate

# 科学推理
from src.specialized_cot.scienceqa_cot import scienceqa_generate

# 实验性VoT
from src.experimental.vot_prompting import generate_vot
```

### 3. 命令行使用
```bash
# 基础推理
python src/basic_cot/infer_cot.py --image image.jpg --question "描述图片"

# 高级推理
python src/advanced_cot/advanced_cot.py --image image.jpg --question "分析图片" --enable-advanced

# 科学推理
python src/specialized_cot/scienceqa_cot.py --image diagram.jpg --question "解释现象"

# VoT推理
python src/experimental/vot_prompting.py --task_type visual_navigation --question "导航路径"
```

## 🔧 技术架构

### 依赖关系
```
experimental/  ←── advanced_cot/  ←── basic_cot/
     ↓               ↓                ↓
specialized_cot/  ←── core/
```

### 核心特性
- **模块化设计**: 每个包独立，便于维护和扩展
- **渐进式复杂度**: 从基础到高级，满足不同需求
- **专门化优化**: 针对特定任务进行优化
- **实验性支持**: 支持前沿研究和新方法

## 📊 性能对比

| 包类型 | 推理速度 | 内存使用 | 准确率 | 适用场景 |
|--------|----------|----------|--------|----------|
| basic_cot | 快 | 低 | 中等 | 日常使用 |
| advanced_cot | 中等 | 中等 | 高 | 复杂任务 |
| specialized_cot | 中等 | 中等 | 很高 | 特定领域 |
| experimental | 慢 | 高 | 未知 | 研究探索 |

## 🔮 未来扩展

### 计划添加的包
- **distributed_cot/**: 分布式CoT推理
- **real_time_cot/**: 实时CoT推理
- **multilingual_cot/**: 多语言CoT支持
- **domain_specific/**: 领域特定CoT

### 优化方向
- **性能优化**: 更快的推理速度
- **内存优化**: 更低的内存占用
- **准确率提升**: 更高的推理准确率
- **易用性**: 更简单的使用方式

## 📝 维护说明

### 代码规范
- 每个包都有独立的 `__init__.py`
- 统一的错误处理机制
- 完整的文档字符串
- 类型注解支持

### 测试策略
- 单元测试覆盖核心功能
- 集成测试验证包间协作
- 性能测试确保效率
- 兼容性测试支持多平台

---

**总结**: 分包整理已完成，系统现在具有清晰的模块化结构，便于维护、扩展和使用。每个包都有明确的功能定位，用户可以根据需求选择合适的实现。
