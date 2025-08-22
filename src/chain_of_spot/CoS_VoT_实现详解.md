# CoS + VoT 实现详解

## 🎯 核心创新理念

CoS + VoT 的核心创新在于将Chain-of-Spot的ROI定位能力与Visualization-of-Thought的可视化推理能力相结合，创造出"可视化交互式推理"方法。

### 传统方法的局限
- **CoS**: 只能定位ROI，缺乏可视化推理过程
- **VoT**: 只能可视化推理，缺乏精确的ROI定位

### CoS + VoT 的优势
- **精确ROI定位** + **可视化推理过程**
- **多步骤细化** + **空间可视化**
- **动态调整** + **轨迹记录**

## 🏗️ 架构设计

### 1. 核心数据结构

```python
@dataclass
class VisualROI:
    """可视化ROI结构"""
    bbox: BoundingBox          # ROI边界框
    confidence: float          # 置信度
    visualization: str         # 文本形式的可视化
    reasoning: str            # 推理过程
    step_id: int              # 步骤ID

@dataclass
class CoSVoTResponse:
    """CoS+VoT混合响应"""
    final_roi: BoundingBox                    # 最终ROI
    final_answer: str                         # 最终答案
    visual_trajectory: List[VisualROI]        # 可视化轨迹
    reasoning_trace: List[str]                # 推理轨迹
    confidence: float                         # 置信度
    spatial_visualization: str               # 空间可视化
```

### 2. 主要组件

#### SpatialVisualizer (空间可视化器)
```python
class SpatialVisualizer:
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
        self.grid_chars = {
            'empty': '·',      # 空区域
            'target': '★',     # 目标中心
            'roi': '█',        # ROI区域
            'context': '░',    # 上下文
            'boundary': '│'    # 边界
        }
```

#### CoSVoTModel (混合模型)
```python
class CoSVoTModel(ChainOfSpotModel):
    def __init__(self, base_model, processor, device: str = "auto"):
        super().__init__(base_model, processor, device)
        self.spatial_visualizer = SpatialVisualizer()
```

## 🔄 实现流程

### 步骤1: 多步骤ROI细化

```python
def _multi_step_roi_refinement(self, image, question, max_steps=3):
    visual_rois = []
    
    for step in range(max_steps):
        # 构建当前步骤的指令
        if step == 0:
            instruction = self.vot_instruction_1.replace("<Q>", question)
        else:
            # 基于前一步结果进行细化
            prev_roi = visual_rois[-1]
            instruction = f"<Img> Previous ROI: {prev_roi.bbox.to_string()} "
                        f"Confidence: {prev_roi.confidence:.2f}\n"
                        f"Question: {question}\n"
                        f"Please refine the ROI based on the previous result."
        
        # 调用模型
        response = self._call_model([image], instruction)
        
        # 提取坐标和可视化
        bbox, visualization = self._extract_coords_and_visualization(response)
        
        # 计算置信度
        confidence = self._calculate_step_confidence(response, bbox, step)
        
        # 创建VisualROI
        visual_roi = VisualROI(
            bbox=bbox,
            confidence=confidence,
            visualization=visualization,
            reasoning=response,
            step_id=step + 1
        )
        
        visual_rois.append(visual_roi)
        
        # 如果置信度足够高，提前停止
        if confidence > 0.8:
            break
    
    return visual_rois
```

### 步骤2: 坐标和可视化提取

```python
def _extract_coords_and_visualization(self, response: str):
    bbox = None
    visualization = ""
    
    # 提取坐标
    coords_match = re.search(r'COORDS:\[([\d.]+),([\d.]+),([\d.]+),([\d.]+)\]', response)
    if coords_match:
        coords = [float(x) for x in coords_match.groups()]
        bbox = BoundingBox(x0=coords[0], x1=coords[1], y0=coords[2], y1=coords[3])
    
    # 提取可视化
    visual_match = re.search(r'VISUAL:(.*?)(?=COORDS:|$)', response, re.DOTALL)
    if visual_match:
        visualization = visual_match.group(1).strip()
    
    return bbox, visualization
```

### 步骤3: 置信度计算

```python
def _calculate_step_confidence(self, response: str, bbox: BoundingBox, step: int):
    confidence = 0.5  # 基础置信度
    
    # 基于响应质量
    if "COORDS:" in response and "VISUAL:" in response:
        confidence += 0.2
    
    # 基于ROI大小
    roi_area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)
    if 0.1 <= roi_area <= 0.5:
        confidence += 0.1
    elif roi_area < 0.1:
        confidence -= 0.1
    
    # 基于步骤数（越后面的步骤越可信）
    confidence += step * 0.05
    
    return min(confidence, 1.0)
```

### 步骤4: 动态ROI调整

```python
def _dynamic_roi_adjustment(self, visual_rois: List[VisualROI]):
    if not visual_rois:
        return BoundingBox(x0=0.25, x1=0.75, y0=0.25, y1=0.75)
    
    # 基于置信度加权平均
    total_weight = 0
    weighted_x0 = 0
    weighted_x1 = 0
    weighted_y0 = 0
    weighted_y1 = 0
    
    for roi in visual_rois:
        weight = roi.confidence ** 2  # 平方权重
        total_weight += weight
        
        weighted_x0 += roi.bbox.x0 * weight
        weighted_x1 += roi.bbox.x1 * weight
        weighted_y0 += roi.bbox.y0 * weight
        weighted_y1 += roi.bbox.y1 * weight
    
    if total_weight > 0:
        final_bbox = BoundingBox(
            x0=weighted_x0 / total_weight,
            x1=weighted_x1 / total_weight,
            y0=weighted_y0 / total_weight,
            y1=weighted_y1 / total_weight
        )
    else:
        final_bbox = visual_rois[-1].bbox
    
    return final_bbox
```

## 🎨 空间可视化实现

### 1. 单ROI可视化

```python
def create_spatial_grid(self, bbox: BoundingBox, target_desc: str = "目标"):
    grid = []
    for y in range(self.grid_size):
        row = []
        for x in range(self.grid_size):
            # 归一化坐标
            norm_x = x / self.grid_size
            norm_y = y / self.grid_size
            
            if bbox.x0 <= norm_x <= bbox.x1 and bbox.y0 <= norm_y <= bbox.y1:
                if norm_x == (bbox.x0 + bbox.x1) / 2 and norm_y == (bbox.y0 + bbox.y1) / 2:
                    row.append(self.grid_chars['target'])  # 目标中心
                else:
                    row.append(self.grid_chars['roi'])     # ROI区域
            else:
                row.append(self.grid_chars['empty'])       # 空区域
        grid.append(''.join(row))
    
    # 添加边界和标签
    result = f"空间可视化 - {target_desc}:\n"
    result += "┌" + "─" * self.grid_size + "┐\n"
    for row in grid:
        result += "│" + row + "│\n"
    result += "└" + "─" * self.grid_size + "┘\n"
    result += f"坐标: [{bbox.x0:.2f},{bbox.x1:.2f},{bbox.y0:.2f},{bbox.y1:.2f}]\n"
    
    return result
```

### 2. 多ROI演化可视化

```python
def create_multi_roi_visualization(self, rois: List[VisualROI]):
    grid = []
    for y in range(self.grid_size):
        row = []
        for x in range(self.grid_size):
            norm_x = x / self.grid_size
            norm_y = y / self.grid_size
            
            # 检查是否在任何ROI内
            in_roi = False
            for i, roi in enumerate(rois):
                if roi.bbox.x0 <= norm_x <= roi.bbox.x1 and roi.bbox.y0 <= norm_y <= roi.bbox.y1:
                    row.append(str(i + 1))  # 使用数字标识不同ROI
                    in_roi = True
                    break
            
            if not in_roi:
                row.append(self.grid_chars['empty'])
        grid.append(''.join(row))
    
    result = "多ROI演化可视化:\n"
    result += "┌" + "─" * self.grid_size + "┐\n"
    for row in grid:
        result += "│" + row + "│\n"
    result += "└" + "─" * self.grid_size + "┘\n"
    
    # 添加图例
    for i, roi in enumerate(rois):
        result += f"ROI{i+1}: 置信度={roi.confidence:.2f}, 步骤={roi.step_id}\n"
    
    return result
```

## 🔧 指令模板设计

### 1. 初始ROI定位指令

```python
self.vot_instruction_1 = (
    "<Img> To answer the question: <Q>, "
    "please identify the region of interest and provide a spatial visualization. "
    "Return coordinates as [x0,x1,y0,y1] and create a text-based spatial grid. "
    "Format: COORDS:[x0,x1,y0,y1] VISUAL:[grid visualization]"
)
```

### 2. ROI细化指令

```python
instruction = (
    f"<Img> Previous ROI: {prev_roi.bbox.to_string()} "
    f"Confidence: {prev_roi.confidence:.2f}\n"
    f"Question: {question}\n"
    "Please refine the ROI based on the previous result. "
    "Provide updated coordinates and visualization."
)
```

### 3. 最终分析指令

```python
self.vot_instruction_2 = (
    "The region of interest is <ROI Img>. "
    "Based on this focused region and the spatial context, "
    "please provide a detailed answer to: <Q>. "
    "Include spatial reasoning in your response."
)
```

## 📊 输出示例

### 空间可视化输出
```
空间可视化 - 目标:
┌────────┐
│·····█··│
│····█★█·│
│·····█··│
│·······│
│·······│
│·······│
│·······│
│·······│
└────────┘
坐标: [0.50,0.75,0.25,0.50]
```

### 多ROI演化可视化
```
多ROI演化可视化:
┌────────┐
│·····1··│
│····1★1·│
│·····1··│
│·······│
│·······│
│·······│
│·······│
│·······│
└────────┘
ROI1: 置信度=0.75, 步骤=1
ROI2: 置信度=0.85, 步骤=2
ROI3: 置信度=0.92, 步骤=3
```

## 🚀 使用方法

### 1. 基本使用

```python
from src.chain_of_spot.cos_vot_hybrid import cos_vot_generate

result = cos_vot_generate(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    image_path="image.jpg",
    question="红色汽车在哪里？",
    device="mps",
    max_roi_steps=3,
    save_visualization=True
)
```

### 2. 命令行使用

```bash
python src/chain_of_spot/cos_vot_hybrid.py \
  --image image.jpg \
  --question "红色汽车在哪里？" \
  --device mps \
  --max-roi-steps 3 \
  --save-viz
```

## 🎯 创新特点总结

### 1. **多步骤ROI细化**
- 不是一次性定位，而是逐步优化
- 每一步都基于前一步的结果
- 支持提前停止机制

### 2. **空间可视化**
- 文本形式的网格可视化
- 支持单ROI和多ROI可视化
- 直观显示ROI位置和演化

### 3. **动态调整机制**
- 基于置信度的加权平均
- 自适应ROI调整
- 多步骤结果融合

### 4. **可视化轨迹记录**
- 记录每一步的ROI变化
- 保存推理过程和可视化
- 完整的演化历史

### 5. **置信度评估**
- 多因素置信度计算
- 基于响应质量、ROI大小、步骤数
- 动态权重调整

## 🔬 技术优势

1. **可解释性强**: 可视化推理过程
2. **精度高**: 多步骤细化提高定位精度
3. **鲁棒性好**: 动态调整机制增强稳定性
4. **交互性强**: 支持用户查看推理轨迹
5. **扩展性好**: 易于集成其他方法

---

**CoS + VoT 通过可视化交互式推理，实现了ROI定位和推理过程的可视化，大大提升了多模态推理的可解释性和准确性！** 🚀
