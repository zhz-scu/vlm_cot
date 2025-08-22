#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU支持测试脚本

测试所有MCOT方法在NPU上的兼容性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
import argparse
import time

# 导入所有MCOT方法
from src.basic_cot.infer_cot import infer_cot_generate
from src.advanced_cot.advanced_cot import advanced_cot_generate
from src.advanced_cot.enhanced_cot import enhanced_mcot_generate
from src.advanced_cot.advanced_mcot import advanced_mcot_generate
from src.chain_of_spot.cos_vot_npu import cos_vot_npu_generate
from src.core.npu_utils import print_npu_status, auto_select_device


def test_device_detection():
    """测试设备检测"""
    print("=== 设备检测测试 ===")
    
    devices = ["auto", "npu", "cuda", "mps", "cpu"]
    for device in devices:
        selected = auto_select_device(device)
        print(f"输入: {device} -> 选择: {selected}")
    
    print()


def test_npu_availability():
    """测试NPU可用性"""
    print("=== NPU可用性测试 ===")
    print_npu_status()
    print()


def test_method_compatibility(method_name: str, method_func, test_image: str, test_question: str, device: str):
    """测试方法兼容性"""
    print(f"=== 测试 {method_name} ===")
    
    try:
        start_time = time.time()
        
        if method_name == "CoS+VoT NPU":
            result = method_func(
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                image_path=test_image,
                question=test_question,
                device=device,
                dtype_str="fp16"
            )
        else:
            result = method_func(
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                images=[test_image],
                question=test_question,
                device=device,
                dtype_str="fp16"
            )
        
        end_time = time.time()
        
        print(f"✅ {method_name} 成功")
        print(f"   设备: {device}")
        print(f"   耗时: {end_time - start_time:.2f}秒")
        
        if method_name == "CoS+VoT NPU":
            print(f"   答案: {result['final_answer'][:100]}...")
            print(f"   置信度: {result['confidence']:.3f}")
        else:
            if 'answer' in result:
                print(f"   答案: {result['answer'][:100]}...")
            elif 'final_answer' in result:
                print(f"   答案: {result['final_answer'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ {method_name} 失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="NPU支持测试")
    parser.add_argument("--image", type=str, default="test_simple.png", help="测试图像")
    parser.add_argument("--question", type=str, default="这个图片中有什么？", help="测试问题")
    parser.add_argument("--device", type=str, default="auto", help="测试设备")
    parser.add_argument("--methods", type=str, nargs="+", 
                       default=["basic_cot", "advanced_cot", "enhanced_cot", "advanced_mcot", "cos_vot_npu"],
                       help="要测试的方法")
    
    args = parser.parse_args()
    
    # 检查测试图像
    if not os.path.exists(args.image):
        print(f"错误: 测试图像不存在: {args.image}")
        sys.exit(1)
    
    print("🧠 NPU支持测试开始")
    print(f"测试图像: {args.image}")
    print(f"测试问题: {args.question}")
    print(f"测试设备: {args.device}")
    print()
    
    # 测试设备检测
    test_device_detection()
    
    # 测试NPU可用性
    test_npu_availability()
    
    # 方法映射
    method_map = {
        "basic_cot": ("基础CoT", infer_cot_generate),
        "advanced_cot": ("高级CoT", advanced_cot_generate),
        "enhanced_cot": ("增强CoT", enhanced_mcot_generate),
        "advanced_mcot": ("高级MCOT", advanced_mcot_generate),
        "cos_vot_npu": ("CoS+VoT NPU", cos_vot_npu_generate)
    }
    
    # 测试方法兼容性
    success_count = 0
    total_count = 0
    
    for method_key in args.methods:
        if method_key in method_map:
            method_name, method_func = method_map[method_key]
            total_count += 1
            
            if test_method_compatibility(method_name, method_func, args.image, args.question, args.device):
                success_count += 1
            
            print()
        else:
            print(f"警告: 未知方法 {method_key}")
    
    # 总结
    print("=== 测试总结 ===")
    print(f"成功: {success_count}/{total_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 所有方法都支持NPU!")
    else:
        print("⚠️  部分方法需要进一步适配")


if __name__ == "__main__":
    main()
