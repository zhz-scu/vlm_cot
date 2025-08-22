#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain-of-Spot + VoT 混合方法: 可视化交互式推理

创新点：
1. 结合CoS的ROI定位和VoT的可视化推理
2. 在ROI定位过程中加入空间可视化
3. 多步骤ROI细化与可视化验证
4. 动态ROI调整机制
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import re
import json

from cos_model import ChainOfSpotModel, BoundingBox, CoSResponse


@dataclass
class VisualROI:
    """可视化ROI结构"""
    bbox: BoundingBox
    confidence: float
    visualization: str  # 文本形式的可视化
    reasoning: str     # 推理过程
    step_id: int       # 步骤ID


@dataclass
class CoSVoTResponse:
    """CoS+VoT混合响应"""
    final_roi: BoundingBox
    final_answer: str
    visual_trajectory: List[VisualROI]  # 可视化轨迹
    reasoning_trace: List[str]
    confidence: float
    spatial_visualization: str  # 最终的空间可视化


class SpatialVisualizer:
    """空间可视化器"""
    
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
        self.grid_chars = {
            'empty': '·',
            'target': '★',
            'roi': '█',
            'context': '░',
            'boundary': '│'
        }
    
    def create_spatial_grid(self, bbox: BoundingBox, target_desc: str = "目标") -> str:
        """创建空间网格可视化"""
        grid = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                # 归一化坐标
                norm_x = x / self.grid_size
                norm_y = y / self.grid_size
                
                if bbox.x0 <= norm_x <= bbox.x1 and bbox.y0 <= norm_y <= bbox.y1:
                    if norm_x == (bbox.x0 + bbox.x1) / 2 and norm_y == (bbox.y0 + bbox.y1) / 2:
                        row.append(self.grid_chars['target'])
                    else:
                        row.append(self.grid_chars['roi'])
                else:
                    row.append(self.grid_chars['empty'])
            grid.append(''.join(row))
        
        # 添加边界和标签
        result = f"空间可视化 - {target_desc}:\n"
        result += "┌" + "─" * self.grid_size + "┐\n"
        for row in grid:
            result += "│" + row + "│\n"
        result += "└" + "─" * self.grid_size + "┘\n"
        result += f"坐标: [{bbox.x0:.2f},{bbox.x1:.2f},{bbox.y0:.2f},{bbox.y1:.2f}]\n"
        
        return result
    
    def create_multi_roi_visualization(self, rois: List[VisualROI]) -> str:
        """创建多ROI可视化"""
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


class CoSVoTModel(ChainOfSpotModel):
    """Chain-of-Spot + VoT 混合模型"""
    
    def __init__(self, base_model, processor, device: str = "auto"):
        super().__init__(base_model, processor, device)
        self.spatial_visualizer = SpatialVisualizer()
        
        # 改进的指令模板
        self.vot_instruction_1 = (
            "<Img> To answer the question: <Q>, "
            "please identify the region of interest and provide a spatial visualization. "
            "Return coordinates as [x0,x1,y0,y1] and create a text-based spatial grid. "
            "Format: COORDS:[x0,x1,y0,y1] VISUAL:[grid visualization]"
        )
        
        self.vot_instruction_2 = (
            "The region of interest is <ROI Img>. "
            "Based on this focused region and the spatial context, "
            "please provide a detailed answer to: <Q>. "
            "Include spatial reasoning in your response."
        )
    
    def _extract_coords_and_visualization(self, response: str) -> Tuple[Optional[BoundingBox], str]:
        """提取坐标和可视化信息"""
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
    
    def _multi_step_roi_refinement(self, image: Image.Image, question: str, 
                                 max_steps: int = 3) -> List[VisualROI]:
        """多步骤ROI细化"""
        visual_rois = []
        current_bbox = None
        
        for step in range(max_steps):
            # 构建当前步骤的指令
            if step == 0:
                instruction = self.vot_instruction_1.replace("<Q>", question)
            else:
                # 基于前一步结果进行细化
                prev_roi = visual_rois[-1]
                instruction = (
                    f"<Img> Previous ROI: {prev_roi.bbox.to_string()} "
                    f"Confidence: {prev_roi.confidence:.2f}\n"
                    f"Question: {question}\n"
                    "Please refine the ROI based on the previous result. "
                    "Provide updated coordinates and visualization."
                )
            
            # 调用模型
            response = self._call_model([image], instruction)
            
            # 提取结果
            bbox, visualization = self._extract_coords_and_visualization(response)
            
            if bbox is None:
                # 使用启发式方法
                bbox = self._heuristic_roi_extraction(image, question)
                visualization = self.spatial_visualizer.create_spatial_grid(bbox, f"步骤{step+1}")
            
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
            current_bbox = bbox
            
            # 如果置信度足够高，提前停止
            if confidence > 0.8:
                break
        
        return visual_rois
    
    def _calculate_step_confidence(self, response: str, bbox: BoundingBox, step: int) -> float:
        """计算步骤置信度"""
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
    
    def _dynamic_roi_adjustment(self, visual_rois: List[VisualROI]) -> BoundingBox:
        """动态ROI调整"""
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
    
    def visual_interactive_reasoning(self, image: Image.Image, question: str) -> CoSVoTResponse:
        """
        可视化交互式推理 - 主要方法
        """
        reasoning_trace = []
        
        # 步骤1: 多步骤ROI细化
        reasoning_trace.append("开始多步骤ROI细化...")
        visual_rois = self._multi_step_roi_refinement(image, question)
        reasoning_trace.append(f"完成{len(visual_rois)}步ROI细化")
        
        # 步骤2: 动态ROI调整
        final_roi = self._dynamic_roi_adjustment(visual_rois)
        reasoning_trace.append(f"最终ROI: {final_roi.to_string()}")
        
        # 步骤3: 裁剪ROI图像
        roi_image = self.image_cropper.crop_image(image, final_roi)
        reasoning_trace.append("ROI图像裁剪完成")
        
        # 步骤4: 基于ROI的详细分析
        instruction_2 = self.vot_instruction_2.replace("<Q>", question)
        reasoning_trace.append("开始详细分析...")
        
        final_answer = self._call_model([image, roi_image], instruction_2)
        reasoning_trace.append("详细分析完成")
        
        # 步骤5: 生成最终空间可视化
        spatial_viz = self.spatial_visualizer.create_multi_roi_visualization(visual_rois)
        
        # 计算最终置信度
        final_confidence = np.mean([roi.confidence for roi in visual_rois])
        
        return CoSVoTResponse(
            final_roi=final_roi,
            final_answer=final_answer,
            visual_trajectory=visual_rois,
            reasoning_trace=reasoning_trace,
            confidence=final_confidence,
            spatial_visualization=spatial_viz
        )


def cos_vot_generate(
    model_id: str,
    image_path: str,
    question: str,
    device: str = "auto",
    dtype_str: str = "auto",
    max_new_tokens: int = 512,
    max_roi_steps: int = 3,
    seed: int = None,
    save_visualization: bool = False,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    CoS+VoT混合方法生成函数
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # 加载模型
    device = "mps" if device == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else device
    torch_dtype = torch.float16 if dtype_str == "auto" and device == "mps" else torch.float32
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except:
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # 初始化混合模型
    cos_vot_model = CoSVoTModel(model, processor, device)
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 执行可视化交互式推理
    response = cos_vot_model.visual_interactive_reasoning(image, question)
    
    # 保存可视化结果
    if save_visualization:
        # 创建可视化图像
        viz_image = cos_vot_model.image_cropper.visualize_roi(image, response.final_roi)
        viz_path = f"{output_dir}/cos_vot_visualization.png"
        viz_image.save(viz_path)
        
        # 保存空间可视化文本
        viz_text_path = f"{output_dir}/spatial_visualization.txt"
        with open(viz_text_path, 'w', encoding='utf-8') as f:
            f.write(response.spatial_visualization)
    
    # 构建返回结果
    result = {
        "method": "CoS+VoT混合方法",
        "question": question,
        "image_path": image_path,
        "final_roi": response.final_roi.to_string(),
        "final_answer": response.final_answer,
        "confidence": response.confidence,
        "roi_steps": len(response.visual_trajectory),
        "visual_trajectory": [
            {
                "step": roi.step_id,
                "bbox": roi.bbox.to_string(),
                "confidence": roi.confidence,
                "visualization": roi.visualization,
                "reasoning": roi.reasoning
            }
            for roi in response.visual_trajectory
        ],
        "reasoning_trace": response.reasoning_trace,
        "spatial_visualization": response.spatial_visualization,
        "innovation_features": [
            "多步骤ROI细化",
            "空间可视化推理",
            "动态ROI调整",
            "置信度加权平均",
            "可视化轨迹记录"
        ]
    }
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CoS+VoT混合方法测试")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--question", required=True, help="问题")
    parser.add_argument("--device", default="auto", help="设备")
    parser.add_argument("--dtype", default="auto", help="数据类型")
    parser.add_argument("--max-roi-steps", type=int, default=3, help="最大ROI步骤数")
    parser.add_argument("--save-viz", action="store_true", help="保存可视化")
    
    args = parser.parse_args()
    
    result = cos_vot_generate(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        image_path=args.image,
        question=args.question,
        device=args.device,
        dtype_str=args.dtype,
        max_roi_steps=args.max_roi_steps,
        save_visualization=args.save_viz
    )
    
    print("=" * 80)
    print("🔬 CoS+VoT 混合方法结果")
    print("=" * 80)
    print(f"问题: {result['question']}")
    print(f"最终ROI: {result['final_roi']}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"ROI步骤数: {result['roi_steps']}")
    print(f"最终答案: {result['final_answer']}")
    
    print("\n📊 可视化轨迹:")
    for step in result['visual_trajectory']:
        print(f"步骤{step['step']}: ROI={step['bbox']}, 置信度={step['confidence']:.3f}")
    
    print("\n🎨 空间可视化:")
    print(result['spatial_visualization'])
    
    print("\n✨ 创新特性:")
    for feature in result['innovation_features']:
        print(f"- {feature}")
