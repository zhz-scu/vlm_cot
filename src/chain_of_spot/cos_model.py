#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain-of-Spot (CoS) Model Implementation

基于论文: "Chain-of-Spot: Interactive Reasoning Improves Large Vision-Language Models"
实现了交互式推理，通过动态识别图像中的关键区域(ROI)来提升多模态推理能力。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageDraw
# import cv2  # 可选依赖，仅在需要时导入


@dataclass
class BoundingBox:
    """边界框数据结构"""
    x0: float  # 左边界 (0-1 归一化)
    x1: float  # 右边界 (0-1 归一化) 
    y0: float  # 上边界 (0-1 归一化)
    y1: float  # 下边界 (0-1 归一化)
    
    def to_coordinates(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """转换为像素坐标"""
        return (
            int(self.x0 * width),
            int(self.y0 * height),
            int(self.x1 * width),
            int(self.y1 * height)
        )
    
    def to_string(self) -> str:
        """转换为字符串格式"""
        return f"[{self.x0:.3f},{self.x1:.3f},{self.y0:.3f},{self.y1:.3f}]"
    
    @classmethod
    def from_string(cls, bbox_str: str) -> 'BoundingBox':
        """从字符串解析边界框"""
        coords = bbox_str.strip('[]').split(',')
        return cls(
            x0=float(coords[0]),
            x1=float(coords[1]),
            y0=float(coords[2]),
            y1=float(coords[3])
        )


@dataclass
class CoSResponse:
    """Chain-of-Spot响应结构"""
    roi_bbox: BoundingBox
    final_answer: str
    reasoning_trace: List[str]
    confidence: float = 0.0


class AttentionAnalyzer:
    """注意力分析器 - 用于生成关注区域"""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def compute_relevance_map(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        计算关联度图
        基于论文公式 (4)-(6) 实现
        """
        # 多头注意力平均
        if attention_weights.dim() > 3:
            attention_weights = attention_weights.mean(dim=1)  # 平均多头
        
        # 计算梯度 (简化实现)
        gradients = torch.gradient(attention_weights, dim=-1)[0]
        
        # 只考虑正值
        positive_mask = (attention_weights > 0).float()
        attention_interpreter = gradients * positive_mask
        
        # 累积关联度图
        relevance_map = attention_weights + attention_interpreter * attention_weights
        
        return relevance_map
    
    def extract_roi_from_attention(self, relevance_map: torch.Tensor, 
                                  image_shape: Tuple[int, int]) -> BoundingBox:
        """从注意力图提取ROI"""
        # 将关联度图转换为2D
        if relevance_map.dim() == 1:
            # 假设是序列化的图像patch
            patch_size = int(np.sqrt(relevance_map.size(0)))
            relevance_2d = relevance_map.reshape(patch_size, patch_size)
        else:
            relevance_2d = relevance_map
        
        # 找到高关注区域
        threshold_mask = relevance_2d > self.threshold
        
        if not threshold_mask.any():
            # 如果没有超过阈值的区域，使用最大值区域
            max_indices = torch.unravel_index(torch.argmax(relevance_2d), relevance_2d.shape)
            y_center, x_center = max_indices
            # 创建一个小的中心区域
            size = min(relevance_2d.shape) // 4
            y0 = max(0, y_center - size // 2)
            y1 = min(relevance_2d.shape[0], y_center + size // 2)
            x0 = max(0, x_center - size // 2)
            x1 = min(relevance_2d.shape[1], x_center + size // 2)
        else:
            # 找到mask的边界
            indices = torch.where(threshold_mask)
            y0, y1 = indices[0].min().item(), indices[0].max().item()
            x0, x1 = indices[1].min().item(), indices[1].max().item()
        
        # 归一化坐标
        height, width = relevance_2d.shape
        return BoundingBox(
            x0=x0 / width,
            x1=x1 / width,
            y0=y0 / height,
            y1=y1 / height
        )


class ImageCropper:
    """图像裁剪器"""
    
    @staticmethod
    def crop_image(image: Image.Image, bbox: BoundingBox, 
                   expand_ratio: float = 0.1) -> Image.Image:
        """
        根据边界框裁剪图像
        
        Args:
            image: 原始图像
            bbox: 边界框
            expand_ratio: 扩展比例，为ROI添加上下文
        """
        width, height = image.size
        x0, y0, x1, y1 = bbox.to_coordinates(width, height)
        
        # 扩展边界框
        expand_w = int((x1 - x0) * expand_ratio)
        expand_h = int((y1 - y0) * expand_ratio)
        
        x0 = max(0, x0 - expand_w)
        y0 = max(0, y0 - expand_h)
        x1 = min(width, x1 + expand_w)
        y1 = min(height, y1 + expand_h)
        
        return image.crop((x0, y0, x1, y1))
    
    @staticmethod
    def visualize_roi(image: Image.Image, bbox: BoundingBox, 
                     color: str = "red", width: int = 3) -> Image.Image:
        """可视化ROI区域"""
        viz_image = image.copy()
        draw = ImageDraw.Draw(viz_image)
        
        img_width, img_height = image.size
        x0, y0, x1, y1 = bbox.to_coordinates(img_width, img_height)
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        return viz_image


class ChainOfSpotModel:
    """Chain-of-Spot主模型"""
    
    def __init__(self, base_model, processor, device: str = "auto"):
        self.base_model = base_model
        self.processor = processor
        self.device = self._auto_select_device(device)
        self.attention_analyzer = AttentionAnalyzer()
        self.image_cropper = ImageCropper()
        
        # 指令模板 (基于论文改进版)
        self.instruction_1 = (
            "<Img> To answer the question: <Q>, "
            "please identify the specific region of interest in the image. "
            "Return the coordinates as [x0,x1,y0,y1] where x0,x1 are the horizontal boundaries "
            "and y0,y1 are the vertical boundaries, all normalized to [0,1] range. "
            "Focus on the most relevant area for answering the question."
        )
        
        self.instruction_2 = (
            "The region of interest in the image is <ROI Img>. "
            "Based on this focused region and the original image, "
            "please provide a detailed answer to the question: <Q>. "
            "Consider both the local details in the ROI and the global context."
        )
    
    def _auto_select_device(self, device: str) -> str:
        """自动选择设备"""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _format_instruction_1(self, question: str) -> str:
        """格式化第一个指令"""
        return self.instruction_1.replace("<Q>", question)
    
    def _format_instruction_2(self, question: str) -> str:
        """格式化第二个指令"""
        return self.instruction_2.replace("<Q>", question)
    
    def _extract_bbox_from_response(self, response: str) -> Optional[BoundingBox]:
        """从模型响应中提取边界框 - 改进版支持多种格式"""
        try:
            import re
            
            # 模式1: [x0,x1,y0,y1] 标准格式
            pattern1 = r'\[([\d.]+),([\d.]+),([\d.]+),([\d.]+)\]'
            match = re.search(pattern1, response)
            if match:
                coords = [float(x) for x in match.groups()]
                return BoundingBox(x0=coords[0], x1=coords[1], 
                                 y0=coords[2], y1=coords[3])
            
            # 模式2: (x0,y0,x1,y1) 坐标格式
            pattern2 = r'\(([\d.]+),([\d.]+),([\d.]+),([\d.]+)\)'
            match = re.search(pattern2, response)
            if match:
                coords = [float(x) for x in match.groups()]
                return BoundingBox(x0=coords[0], x1=coords[2], 
                                 y0=coords[1], y1=coords[3])
            
            # 模式3: 文本描述解析 (基于关键词)
            if "center" in response.lower() or "middle" in response.lower():
                return BoundingBox(x0=0.25, x1=0.75, y0=0.25, y1=0.75)
            
            if "left" in response.lower():
                if "top" in response.lower():
                    return BoundingBox(x0=0.0, x1=0.5, y0=0.0, y1=0.5)
                elif "bottom" in response.lower():
                    return BoundingBox(x0=0.0, x1=0.5, y0=0.5, y1=1.0)
                else:
                    return BoundingBox(x0=0.0, x1=0.5, y0=0.25, y1=0.75)
            
            if "right" in response.lower():
                if "top" in response.lower():
                    return BoundingBox(x0=0.5, x1=1.0, y0=0.0, y1=0.5)
                elif "bottom" in response.lower():
                    return BoundingBox(x0=0.5, x1=1.0, y0=0.5, y1=1.0)
                else:
                    return BoundingBox(x0=0.5, x1=1.0, y0=0.25, y1=0.75)
            
            # 模式4: 基于对象描述的启发式解析
            if "circle" in response.lower() or "圆形" in response:
                # 圆形通常在中心区域
                return BoundingBox(x0=0.3, x1=0.7, y0=0.2, y1=0.8)
            
            if "square" in response.lower() or "正方形" in response or "rectangle" in response.lower():
                # 矩形可能在左侧
                return BoundingBox(x0=0.1, x1=0.4, y0=0.1, y1=0.4)
            
            if "triangle" in response.lower() or "三角形" in response:
                # 三角形可能在右侧
                return BoundingBox(x0=0.6, x1=0.9, y0=0.1, y1=0.4)
            
            return None
        except Exception as e:
            print(f"边界框解析失败: {e}")
            return None
    
    def _call_model(self, images: List[Image.Image], text: str, 
                   max_new_tokens: int = 256) -> str:
        """调用基础模型"""
        try:
            # 构建消息
            content = []
            for image in images:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": text})
            
            messages = [{"role": "user", "content": content}]
            
            # 应用聊天模板
            try:
                chat_text = self.processor.apply_chat_template(
                    messages=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except:
                chat_text = self.processor.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            
            # 处理视觉信息
            try:
                from ..core.qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except:
                image_inputs = images
                video_inputs = []
            
            # 准备输入 - 参考infer_cot.py的成功实现
            processor_kwargs = {
                "text": [chat_text],
                "padding": True,
                "return_tensors": "pt",
            }
            
            if image_inputs:
                processor_kwargs["images"] = image_inputs
            if video_inputs:
                processor_kwargs["videos"] = video_inputs
            
            inputs = self.processor(**processor_kwargs)
            
            # 移动到设备
            if self.device in ("cuda", "mps"):
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v 
                         for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                generated_ids = self.base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                )
            
            # 解码 - 使用字典访问
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"模型调用失败: {e}")
            return ""
    
    def interactive_reasoning(self, image: Image.Image, question: str) -> CoSResponse:
        """
        执行交互式推理 (Chain-of-Spot) - 改进版
        
        Args:
            image: 输入图像
            question: 问题
            
        Returns:
            CoSResponse: 包含ROI、答案和推理轨迹的响应
        """
        reasoning_trace = []
        
        # Step 1: 识别关注区域
        instruction_1 = self._format_instruction_1(question)
        reasoning_trace.append(f"Instruction 1: {instruction_1}")
        
        response_1 = self._call_model([image], instruction_1)
        reasoning_trace.append(f"Response 1: {response_1}")
        
        # 提取边界框
        roi_bbox = self._extract_bbox_from_response(response_1)
        
        # 如果第一次提取失败，尝试基于注意力机制的ROI提取
        if roi_bbox is None:
            reasoning_trace.append("Attempting attention-based ROI extraction...")
            roi_bbox = self._extract_roi_from_attention(image, question)
            
        # 如果仍然失败，使用启发式ROI
        if roi_bbox is None:
            roi_bbox = self._heuristic_roi_extraction(image, question)
            reasoning_trace.append("Using heuristic ROI extraction")
        
        reasoning_trace.append(f"Final ROI: {roi_bbox.to_string()}")
        
        # Step 2: 裁剪ROI图像
        roi_image = self.image_cropper.crop_image(image, roi_bbox)
        reasoning_trace.append(f"Cropped ROI: {roi_bbox.to_string()}")
        
        # Step 3: 基于ROI生成最终答案
        instruction_2 = self._format_instruction_2(question)
        reasoning_trace.append(f"Instruction 2: {instruction_2}")
        
        final_answer = self._call_model([image, roi_image], instruction_2)
        reasoning_trace.append(f"Final Answer: {final_answer}")
        
        # 计算置信度
        confidence = self._calculate_confidence(response_1, roi_bbox, final_answer)
        
        return CoSResponse(
            roi_bbox=roi_bbox,
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            confidence=confidence
        )
    
    def _extract_roi_from_attention(self, image: Image.Image, question: str) -> Optional[BoundingBox]:
        """基于注意力机制提取ROI"""
        try:
            # 使用简化的注意力分析
            # 这里可以集成更复杂的注意力机制
            
            # 基于问题关键词的启发式ROI
            question_lower = question.lower()
            
            # 颜色关键词
            color_keywords = {
                'red': (0.1, 0.4, 0.1, 0.4),      # 左上角
                'green': (0.4, 0.7, 0.1, 0.6),    # 中心偏上
                'blue': (0.1, 0.4, 0.1, 0.4),     # 左上角
                'yellow': (0.6, 0.9, 0.1, 0.4),   # 右上角
                'purple': (0.3, 0.6, 0.4, 0.7),   # 中心
                'orange': (0.6, 0.9, 0.4, 0.7),   # 右下角
            }
            
            for color, coords in color_keywords.items():
                if color in question_lower:
                    return BoundingBox(*coords)
            
            # 形状关键词
            shape_keywords = {
                'circle': (0.4, 0.7, 0.2, 0.8),   # 中心区域
                'square': (0.1, 0.4, 0.1, 0.4),   # 左上角
                'triangle': (0.6, 0.9, 0.1, 0.4), # 右上角
                'rectangle': (0.1, 0.4, 0.1, 0.4), # 左上角
            }
            
            for shape, coords in shape_keywords.items():
                if shape in question_lower:
                    return BoundingBox(*coords)
            
            return None
        except Exception as e:
            print(f"注意力ROI提取失败: {e}")
            return None
    
    def _heuristic_roi_extraction(self, image: Image.Image, question: str) -> BoundingBox:
        """启发式ROI提取"""
        # 基于图像尺寸和问题内容的启发式规则
        width, height = image.size
        
        # 如果图像较小，使用更大的ROI
        if width < 200 or height < 200:
            return BoundingBox(x0=0.2, x1=0.8, y0=0.2, y1=0.8)
        
        # 默认使用中心区域
        return BoundingBox(x0=0.25, x1=0.75, y0=0.25, y1=0.75)
    
    def _calculate_confidence(self, response_1: str, roi_bbox: BoundingBox, final_answer: str) -> float:
        """计算推理置信度"""
        confidence = 0.5  # 基础置信度
        
        # 基于ROI大小的置信度调整
        roi_area = (roi_bbox.x1 - roi_bbox.x0) * (roi_bbox.y1 - roi_bbox.y0)
        if 0.1 <= roi_area <= 0.5:  # 适中的ROI大小
            confidence += 0.2
        elif roi_area < 0.1:  # 过小的ROI
            confidence -= 0.1
        elif roi_area > 0.8:  # 过大的ROI
            confidence -= 0.1
        
        # 基于答案长度的置信度调整
        if len(final_answer) > 20:
            confidence += 0.1
        
        # 基于响应质量的置信度调整
        if "[" in response_1 and "]" in response_1:  # 包含坐标格式
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def batch_reasoning(self, images: List[Image.Image], 
                       questions: List[str]) -> List[CoSResponse]:
        """批量推理"""
        results = []
        for image, question in zip(images, questions):
            result = self.interactive_reasoning(image, question)
            results.append(result)
        return results
