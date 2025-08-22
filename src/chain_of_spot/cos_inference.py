#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain-of-Spot 推理脚本

基于论文实现的交互式推理方法，通过两步推理过程：
1. 识别图像中的关注区域 (ROI)
2. 基于ROI和原图生成最终答案
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_NATIVE_QWEN25_VL = True
except ImportError:
    from transformers import AutoModelForCausalLM
    Qwen2_5_VLForConditionalGeneration = None
    HAS_NATIVE_QWEN25_VL = False

from cos_model import ChainOfSpotModel, CoSResponse


def load_model_and_processor(model_id: str, device: str, dtype_str: str):
    """加载模型和处理器"""
    # 设备选择
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # 数据类型选择
    if dtype_str == "auto":
        torch_dtype = torch.float16 if device == "mps" else torch.float32
    else:
        dtype_mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_mapping.get(dtype_str, torch.float32)
    
    print(f"使用设备: {device}, 数据类型: {torch_dtype}", file=sys.stderr)
    
    # 加载模型
    if HAS_NATIVE_QWEN25_VL:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    print("模型加载成功", file=sys.stderr)
    return model, processor, device


def cos_generate(
    model_id: str,
    image_path: str,
    question: str,
    device: str = "auto",
    dtype_str: str = "auto",
    max_new_tokens: int = 512,
    seed: int = None,
    save_roi_visualization: bool = False,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    Chain-of-Spot 生成函数
    
    Args:
        model_id: 模型ID
        image_path: 图像路径
        question: 问题
        device: 设备
        dtype_str: 数据类型
        max_new_tokens: 最大生成token数
        seed: 随机种子
        save_roi_visualization: 是否保存ROI可视化
        output_dir: 输出目录
        
    Returns:
        Dict: 包含推理结果的字典
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # 加载模型
    model, processor, device = load_model_and_processor(model_id, device, dtype_str)
    
    # 初始化Chain-of-Spot模型
    cos_model = ChainOfSpotModel(model, processor, device)
    
    # 加载图像
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"图像加载成功: {image.size}", file=sys.stderr)
    except Exception as e:
        print(f"图像加载失败: {e}", file=sys.stderr)
        raise
    
    # 执行交互式推理
    print("开始Chain-of-Spot推理...", file=sys.stderr)
    response = cos_model.interactive_reasoning(image, question)
    
    # 保存ROI可视化
    if save_roi_visualization:
        try:
            roi_viz = cos_model.image_cropper.visualize_roi(image, response.roi_bbox)
            viz_path = Path(output_dir) / "roi_visualization.png"
            roi_viz.save(viz_path)
            print(f"ROI可视化已保存: {viz_path}", file=sys.stderr)
        except Exception as e:
            print(f"ROI可视化保存失败: {e}", file=sys.stderr)
    
    # 构建返回结果
    result = {
        "method": "Chain-of-Spot",
        "question": question,
        "image_path": image_path,
        "roi_bbox": response.roi_bbox.to_string(),
        "final_answer": response.final_answer,
        "reasoning_trace": response.reasoning_trace,
        "confidence": response.confidence,
        "interactive_reasoning": True
    }
    
    print("Chain-of-Spot推理完成", file=sys.stderr)
    return result


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Chain-of-Spot: Interactive Reasoning for Large Vision-Language Models"
    )
    
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="模型ID或路径")
    parser.add_argument("--image", type=str, required=True,
                       help="输入图像路径")
    parser.add_argument("--question", type=str, required=True,
                       help="要询问的问题")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="推理设备")
    parser.add_argument("--dtype", type=str, default="auto",
                       choices=["auto", "bf16", "fp16", "fp32"],
                       help="数据类型")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子")
    parser.add_argument("--save-roi-viz", action="store_true",
                       help="保存ROI可视化")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="输出目录")
    parser.add_argument("--json", action="store_true",
                       help="JSON格式输出")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    try:
        result = cos_generate(
            model_id=args.model_id,
            image_path=args.image,
            question=args.question,
            device=args.device,
            dtype_str=args.dtype,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            save_roi_visualization=args.save_roi_viz,
            output_dir=args.output_dir,
        )
        
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("=" * 80)
            print("🔍 Chain-of-Spot 交互式推理结果")
            print("=" * 80)
            print(f"问题: {result['question']}")
            print(f"图像: {result['image_path']}")
            print(f"ROI区域: {result['roi_bbox']}")
            print(f"最终答案: {result['final_answer']}")
            print(f"置信度: {result['confidence']:.3f}")
            
            print("\n📝 推理轨迹:")
            for i, trace in enumerate(result['reasoning_trace'], 1):
                print(f"{i}. {trace}")
            
            print("\n✨ 方法特点:")
            print("- 🎯 交互式推理: 动态识别关注区域")
            print("- 🔍 两步推理: ROI定位 + 细节分析")
            print("- 📊 多粒度特征: 保持原图分辨率的同时关注细节")
            print("- 🚀 性能提升: 在多个多模态数据集上达到SOTA")
            
    except Exception as e:
        print(f"推理失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
