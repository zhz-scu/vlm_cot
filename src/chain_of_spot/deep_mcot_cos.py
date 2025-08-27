#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度MCOT + Chain-of-Spot结合版本

集成YOLO目标检测 + 区域标注 + Basic CoT思维链推理
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageDraw
import re
import json
import time
import cv2
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    HAS_NATIVE_QWEN25_VL = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoProcessor
    Qwen2_5_VLForConditionalGeneration = None
    HAS_NATIVE_QWEN25_VL = False


@dataclass
class DetectedRegion:
    """检测到的区域"""
    bbox: List[float]  # [x0, y0, x1, y1] 归一化坐标
    confidence: float
    label: str
    content: str
    region_id: int


@dataclass
class DeepMCotCoSResponse:
    """深度MCOT+CoS响应"""
    detected_regions: List[DetectedRegion]
    region_annotations: List[str]
    cot_reasoning_steps: List[str]
    final_answer: str
    confidence: float
    detection_time: float
    annotation_time: float
    cot_time: float
    total_time: float


class ObjectDetector:
    """目标检测器"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.initialized = False
    
    def initialize_detector(self, model_type: str = "yolo"):
        """初始化检测器"""
        try:
            if model_type == "yolo":
                try:
                    from ultralytics import YOLO
                    # 优先使用本地权重
                    local_weight = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'yolo', 'yolov8n.pt')
                    local_weight = os.path.abspath(local_weight)
                    if os.path.exists(local_weight):
                        self.model = YOLO(local_weight)
                        print(f"✅ YOLOv8检测器加载本地权重: {local_weight}")
                    else:
                        self.model = YOLO('yolov8n.pt')
                        print("✅ YOLOv8检测器加载默认权重 yolov8n.pt")
                    self.initialized = True
                    print("✅ YOLOv8检测器初始化成功")
                except ImportError:
                    print("⚠️ 未安装ultralytics，使用边缘检测")
                    self._init_edge_detector()
            else:
                self._init_edge_detector()
        except Exception as e:
            print(f"❌ 检测器初始化失败: {e}")
            self.initialized = False
    
    def _init_edge_detector(self):
        """初始化边缘检测器"""
        self.model = "edge_detection"
        self.initialized = True
        print("✅ 边缘检测器初始化成功")
    
    def detect_regions(self, image: Image.Image, min_confidence: float = 0.3) -> List[DetectedRegion]:
        """检测图像中的区域"""
        if not self.initialized:
            self.initialize_detector()
        
        if isinstance(self.model, str) and self.model == "edge_detection":
            return self._edge_detection(image)
        elif hasattr(self.model, 'predict'):
            return self._yolo_detection(image, min_confidence)
        else:
            return self._edge_detection(image)
    
    def _edge_detection(self, image: Image.Image) -> List[DetectedRegion]:
        """边缘检测方法"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        img_height, img_width = gray.shape
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 20:
                continue
            
            x0 = x / img_width
            y0 = y / img_height
            x1 = (x + w) / img_width
            y1 = (y + h) / img_height
            
            area = w * h
            confidence = min(0.8, area / (img_width * img_height) * 10)
            
            regions.append(DetectedRegion(
                bbox=[x0, y0, x1, y1],
                confidence=confidence,
                label=f"region_{i}",
                content="",
                region_id=i
            ))
        
        return regions[:5]
    
    def _yolo_detection(self, image: Image.Image, min_confidence: float) -> List[DetectedRegion]:
        """YOLO检测方法"""
        try:
            results = self.model(image)
            regions = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        confidence = float(box.conf[0])
                        if confidence < min_confidence:
                            continue
                        
                        x0, y0, x1, y1 = box.xyxy[0].cpu().numpy()
                        img_width, img_height = image.size
                        
                        x0_norm = x0 / img_width
                        y0_norm = y0 / img_height
                        x1_norm = x1 / img_width
                        y1_norm = y1 / img_height
                        
                        class_id = int(box.cls[0])
                        label = f"object_{class_id}"
                        
                        regions.append(DetectedRegion(
                            bbox=[x0_norm, y0_norm, x1_norm, y1_norm],
                            confidence=confidence,
                            label=label,
                            content="",
                            region_id=i
                        ))
            
            return regions
        except Exception as e:
            print(f"YOLO检测失败: {e}")
            return self._edge_detection(image)


class RegionAnnotator:
    """区域标注器"""
    
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
    
    def annotate_regions(self, image: Image.Image, regions: List[DetectedRegion]) -> List[str]:
        """标注检测到的区域，返回标注文本列表"""
        annotations = []
        
        for region in regions:
            x0, y0, x1, y1 = region.bbox
            img_width, img_height = image.size
            
            x0_pixel = max(0, min(int(x0 * img_width), img_width))
            y0_pixel = max(0, min(int(y0 * img_height), img_height))
            x1_pixel = max(x0_pixel, min(int(x1 * img_width), img_width))
            y1_pixel = max(y0_pixel, min(int(y1 * img_height), img_height))
            
            if x1_pixel <= x0_pixel or y1_pixel <= y0_pixel:
                continue
            
            try:
                region_image = image.crop((x0_pixel, y0_pixel, x1_pixel, y1_pixel))
                content = self._annotate_single_region(region_image, region.label)
                annotations.append(f"区域{region.region_id}: {content}")
            except Exception as e:
                print(f"区域标注失败: {e}")
                annotations.append(f"区域{region.region_id}: 标注失败")
        
        return annotations
    
    def _annotate_single_region(self, region_image: Image.Image, label: str) -> str:
        """标注单个区域"""
        try:
            prompt = f"请描述这个图像区域中的内容，简洁明了："
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": region_image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            try:
                chat_text = self.processor.apply_chat_template(
                    messages=messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                chat_text = self.processor.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True
                )
            
            inputs = self.processor(
                text=[chat_text],
                images=[region_image],
                padding=True,
                return_tensors="pt",
            )
            
            if self.device in ("cuda", "mps", "npu", "xpu"):
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1,
                )
            
            try:
                trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
                content = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            except Exception:
                content = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            return content.strip()
            
        except Exception as e:
            print(f"区域标注失败: {e}")
            return f"区域内容: {label}"


class BasicCoTReasoner:
    """Basic CoT推理器"""
    
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
    
    def cot_reasoning(self, image: Image.Image, question: str, region_annotations: List[str]) -> List[str]:
        """执行CoT推理，返回推理步骤"""
        
        # 构建CoT提示
        region_text = "\n".join(region_annotations) if region_annotations else "未检测到明显区域"
        
        cot_prompt = f"""请基于以下信息进行逐步推理来回答问题：

检测到的区域信息：
{region_text}

问题：{question}

请按照以下步骤进行推理：
1. 首先分析图像中的主要内容和区域
2. 然后根据问题要求，分析相关区域的特征
3. 最后综合所有信息，给出详细答案

请逐步思考："""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": cot_prompt}
            ]
        }]
        
        try:
            chat_text = self.processor.apply_chat_template(
                messages=messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_text = self.processor.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
        
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        
        if self.device in ("cuda", "mps", "npu", "xpu"):
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
        
        try:
            trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
            reasoning_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception:
            reasoning_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # 解析推理步骤
        reasoning_steps = self._parse_reasoning_steps(reasoning_text)
        return reasoning_steps
    
    def _parse_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """解析推理文本为步骤列表"""
        steps = []
        
        # 尝试按数字编号分割
        lines = reasoning_text.split('\n')
        current_step = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是新的步骤（以数字开头）
            if re.match(r'^\d+[\.\)]', line):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
        
        # 如果没有解析到步骤，将整个文本作为一个步骤
        if not steps:
            steps = [reasoning_text.strip()]
        
        return steps


class DeepMCotCoSModel:
    """深度MCOT+CoS模型"""
    
    def __init__(self, model_id: str, device: str = "auto", detector_type: str = "yolo"):
        self.device = self._auto_select_device(device)
        self.detector_type = detector_type
        
        # 初始化检测器
        self.object_detector = ObjectDetector(self.device)
        self.object_detector.initialize_detector(detector_type)
        
        # 加载模型
        self.model, self.processor = self._load_model(model_id)
        
        # 初始化组件
        self.region_annotator = RegionAnnotator(self.model, self.processor, self.device)
        self.cot_reasoner = BasicCoTReasoner(self.model, self.processor, self.device)
    
    def _auto_select_device(self, device: str) -> str:
        """自动选择设备"""
        if device != "auto":
            return device
        
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return "npu"
        elif torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self, model_id: str):
        """加载模型"""
        torch_dtype = torch.float16 if self.device in ("npu", "mps", "xpu") else torch.float32
        
        if HAS_NATIVE_QWEN25_VL and Qwen2_5_VLForConditionalGeneration is not None:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device in ("cuda", "mps", "npu", "xpu") else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device in ("cuda", "mps", "npu", "xpu") else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        return model, processor
    
    def deep_reasoning(self, image: Image.Image, question: str) -> DeepMCotCoSResponse:
        """深度推理流程"""
        total_start = time.time()
        
        print("🔬 开始深度MCOT+CoS推理...")
        
        # 步骤1: YOLO目标检测
        print("📊 步骤1: YOLO目标检测")
        detection_start = time.time()
        detected_regions = self.object_detector.detect_regions(image, min_confidence=0.3)
        detection_time = time.time() - detection_start
        
        print(f"✅ 检测完成: {len(detected_regions)} 个区域，耗时 {detection_time:.2f}秒")
        
        # 可视化并保存检测结果
        try:
            out_dir = "/test_res"
            os.makedirs(out_dir, exist_ok=True)
            viz_img = _create_visualization(image, detected_regions)
            out_path = os.path.join(out_dir, f"deep_mcot_detection_{self.detector_type}.png")
            viz_img.save(out_path)
            print(f"✅ 检测可视化已保存: {out_path}")
        except Exception as e:
            print(f"⚠️ 检测可视化保存失败: {e}")
        
        # 输出检测结果详情
        if detected_regions:
            print("📋 检测结果明细:")
            for r in detected_regions:
                print(f"  - id={r.region_id}, label={r.label}, conf={r.confidence:.3f}, bbox={r.bbox}")
        else:
            print("📋 检测结果明细: 无检测框")
        
        # 步骤2: 区域标注
        print("\n📝 步骤2: 区域标注")
        annotation_start = time.time()
        region_annotations = self.region_annotator.annotate_regions(image, detected_regions)
        annotation_time = time.time() - annotation_start
        
        print(f"✅ 标注完成: {len(region_annotations)} 个区域，耗时 {annotation_time:.2f}秒")
        for i, annotation in enumerate(region_annotations):
            print(f"  {annotation}")
        
        # 步骤3: Basic CoT推理
        print("\n🧠 步骤3: Basic CoT推理")
        cot_start = time.time()
        cot_steps = self.cot_reasoner.cot_reasoning(image, question, region_annotations)
        cot_time = time.time() - cot_start
        
        print(f"✅ CoT推理完成: {len(cot_steps)} 个步骤，耗时 {cot_time:.2f}秒")
        
        # 提取最终答案（最后一个步骤或整个推理文本）
        final_answer = cot_steps[-1] if cot_steps else "推理失败"
        
        total_time = time.time() - total_start
        
        # 计算置信度
        confidence = np.mean([region.confidence for region in detected_regions]) if detected_regions else 0.5
        
        print(f"\n🎯 深度推理完成，总耗时: {total_time:.2f}秒")
        
        return DeepMCotCoSResponse(
            detected_regions=detected_regions,
            region_annotations=region_annotations,
            cot_reasoning_steps=cot_steps,
            final_answer=final_answer,
            confidence=confidence,
            detection_time=detection_time,
            annotation_time=annotation_time,
            cot_time=cot_time,
            total_time=total_time
        )


def _create_visualization(image: Image.Image, regions: List[DetectedRegion]) -> Image.Image:
    """创建可视化结果"""
    viz_image = image.copy()
    draw = ImageDraw.Draw(viz_image)
    
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    
    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = region.bbox
        
        img_width, img_height = image.size
        x0_pixel = int(x0 * img_width)
        y0_pixel = int(y0 * img_height)
        x1_pixel = int(x1 * img_width)
        y1_pixel = int(y1 * img_height)
        
        draw.rectangle([x0_pixel, y0_pixel, x1_pixel, y1_pixel], outline=color, width=3)
        
        label = f"R{region.region_id}: {region.label}"
        draw.text((x0_pixel, y0_pixel - 20), label, fill=color)
    
    return viz_image


def deep_mcot_cos_generate(
    model_id: str,
    image_path: str,
    question: str,
    device: str = "auto",
    detector_type: str = "yolo",
    save_visualization: bool = False,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """深度MCOT+CoS生成函数"""
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 初始化深度MCOT+CoS模型
    deep_model = DeepMCotCoSModel(model_id, device, detector_type)
    
    # 执行深度推理
    response = deep_model.deep_reasoning(image, question)
    
    # 保存可视化结果
    if save_visualization:
        viz_image = _create_visualization(image, response.detected_regions)
        viz_path = f"{output_dir}/deep_mcot_cos_visualization.png"
        viz_image.save(viz_path)
    
    # 构建返回结果
    result = {
        "method": "Deep MCOT + Chain-of-Spot",
        "device": device,
        "detector_type": detector_type,
        "question": question,
        "image_path": image_path,
        "detected_regions": [
            {
                "region_id": region.region_id,
                "bbox": region.bbox,
                "confidence": region.confidence,
                "label": region.label,
                "content": region.content
            }
            for region in response.detected_regions
        ],
        "region_annotations": response.region_annotations,
        "cot_reasoning_steps": response.cot_reasoning_steps,
        "final_answer": response.final_answer,
        "confidence": response.confidence,
        "timing": {
            "detection_time": response.detection_time,
            "annotation_time": response.annotation_time,
            "cot_time": response.cot_time,
            "total_time": response.total_time
        },
        "improvements": [
            "YOLOv8目标检测",
            "区域级内容标注",
            "Basic CoT思维链推理",
            "多步骤深度分析",
            "可视化检测结果"
        ]
    }
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="深度MCOT+Chain-of-Spot推理")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--question", type=str, required=True, help="问题")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--detector", type=str, default="yolo", choices=["yolo", "edge"], help="检测器类型")
    parser.add_argument("--save-viz", action="store_true", help="保存可视化结果")
    
    args = parser.parse_args()
    
    result = deep_mcot_cos_generate(
        model_id=args.model_id,
        image_path=args.image,
        question=args.question,
        device=args.device,
        detector_type=args.detector,
        save_visualization=args.save_viz
    )
    
    print("\n" + "=" * 80)
    print("🧠 深度MCOT+Chain-of-Spot结果")
    print("=" * 80)
    print(f"设备: {result['device']}")
    print(f"检测器: {result['detector_type']}")
    print(f"问题: {result['question']}")
    print(f"检测区域数: {len(result['detected_regions'])}")
    print(f"检测耗时: {result['timing']['detection_time']:.2f}秒")
    print(f"标注耗时: {result['timing']['annotation_time']:.2f}秒")
    print(f"CoT推理耗时: {result['timing']['cot_time']:.2f}秒")
    print(f"总耗时: {result['timing']['total_time']:.2f}秒")
    print(f"置信度: {result['confidence']:.3f}")
    
    print(f"\n📝 区域标注:")
    for annotation in result['region_annotations']:
        print(f"  {annotation}")
    
    print(f"\n🧠 CoT推理步骤:")
    for i, step in enumerate(result['cot_reasoning_steps'], 1):
        print(f"  步骤{i}: {step}")
    
    print(f"\n💡 最终答案: {result['final_answer']}")
    
    print("\n✨ 改进特性:")
    for improvement in result['improvements']:
        print(f"  - {improvement}")
