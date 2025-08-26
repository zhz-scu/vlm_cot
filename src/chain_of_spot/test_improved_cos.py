#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from improved_cos_model import improved_cos_generate


def test_improved_cos():
    """测试改进的CoS方法"""
    
    # 测试配置
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_path = "test_simple.png"
    question = "图片中有什么？请详细描述。"
    
    print("🔬 开始测试改进的Chain-of-Spot方法")
    print("=" * 60)
    
    # 测试边缘检测器
    print("\n📊 测试边缘检测器...")
    start_time = time.time()
    
    try:
        result_edge = improved_cos_generate(
            model_id=model_id,
            image_path=image_path,
            question=question,
            device="mps",
            detector_type="edge",
            save_visualization=True,
            output_dir="."
        )
        
        edge_time = time.time() - start_time
        
        print("✅ 边缘检测器测试成功")
        print(f"   检测区域数: {len(result_edge['detected_regions'])}")
        print(f"   检测耗时: {result_edge['detection_time']:.2f}秒")
        print(f"   标注耗时: {result_edge['annotation_time']:.2f}秒")
        print(f"   总耗时: {edge_time:.2f}秒")
        print(f"   置信度: {result_edge['confidence']:.3f}")
        
        print("\n📋 检测到的区域:")
        for region in result_edge['detected_regions']:
            print(f"   区域{region['region_id']}: {region['content']} (置信度: {region['confidence']:.3f})")
        
        print(f"\n💡 最终答案: {result_edge['final_answer']}")
        
    except Exception as e:
        print(f"❌ 边缘检测器测试失败: {e}")
        return
    
    # 测试YOLO检测器（如果可用）
    print("\n📊 测试YOLO检测器...")
    start_time = time.time()
    
    try:
        result_yolo = improved_cos_generate(
            model_id=model_id,
            image_path=image_path,
            question=question,
            device="mps",
            detector_type="yolo",
            save_visualization=True,
            output_dir="."
        )
        
        yolo_time = time.time() - start_time
        
        print("✅ YOLO检测器测试成功")
        print(f"   检测区域数: {len(result_yolo['detected_regions'])}")
        print(f"   检测耗时: {result_yolo['detection_time']:.2f}秒")
        print(f"   标注耗时: {result_yolo['annotation_time']:.2f}秒")
        print(f"   总耗时: {yolo_time:.2f}秒")
        print(f"   置信度: {result_yolo['confidence']:.3f}")
        
        print("\n📋 检测到的区域:")
        for region in result_yolo['detected_regions']:
            print(f"   区域{region['region_id']}: {region['content']} (置信度: {region['confidence']:.3f})")
        
        print(f"\n💡 最终答案: {result_yolo['final_answer']}")
        
    except Exception as e:
        print(f"❌ YOLO检测器测试失败: {e}")
    
    # 对比分析
    print("\n" + "=" * 60)
    print("📈 对比分析")
    print("=" * 60)
    
    if 'result_edge' in locals() and 'result_yolo' in locals():
        print(f"边缘检测器:")
        print(f"  - 检测区域数: {len(result_edge['detected_regions'])}")
        print(f"  - 检测耗时: {result_edge['detection_time']:.2f}秒")
        print(f"  - 总耗时: {edge_time:.2f}秒")
        print(f"  - 置信度: {result_edge['confidence']:.3f}")
        
        print(f"\nYOLO检测器:")
        print(f"  - 检测区域数: {len(result_yolo['detected_regions'])}")
        print(f"  - 检测耗时: {result_yolo['detection_time']:.2f}秒")
        print(f"  - 总耗时: {yolo_time:.2f}秒")
        print(f"  - 置信度: {result_yolo['confidence']:.3f}")
        
        # 保存结果
        comparison_result = {
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path,
            "question": question,
            "edge_detector": {
                "regions_count": len(result_edge['detected_regions']),
                "detection_time": result_edge['detection_time'],
                "annotation_time": result_edge['annotation_time'],
                "total_time": edge_time,
                "confidence": result_edge['confidence'],
                "regions": result_edge['detected_regions'],
                "final_answer": result_edge['final_answer']
            },
            "yolo_detector": {
                "regions_count": len(result_yolo['detected_regions']),
                "detection_time": result_yolo['detection_time'],
                "annotation_time": result_yolo['annotation_time'],
                "total_time": yolo_time,
                "confidence": result_yolo['confidence'],
                "regions": result_yolo['detected_regions'],
                "final_answer": result_yolo['final_answer']
            }
        }
        
        with open("improved_cos_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 对比结果已保存到: improved_cos_comparison.json")
    
    print("\n✨ 改进特性总结:")
    improvements = [
        "目标检测算法自动识别ROI区域",
        "模型对每个区域进行内容标注",
        "基于标注信息进行推理",
        "更准确的区域定位和描述",
        "支持多种检测器（边缘检测、YOLO）",
        "可视化检测结果"
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"  {i}. {improvement}")


def test_with_different_questions():
    """使用不同问题测试"""
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_path = "test_simple.png"
    
    questions = [
        "图片中有什么？",
        "请描述图片中的主要物体",
        "图片中有几个物体？",
        "图片中的颜色是什么？",
        "请分析图片的构图"
    ]
    
    print("\n🔍 使用不同问题测试改进的CoS")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 问题 {i}: {question}")
        
        try:
            result = improved_cos_generate(
                model_id=model_id,
                image_path=image_path,
                question=question,
                device="mps",
                detector_type="edge",
                save_visualization=False
            )
            
            print(f"   检测区域数: {len(result['detected_regions'])}")
            print(f"   置信度: {result['confidence']:.3f}")
            print(f"   答案: {result['final_answer'][:100]}...")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")


if __name__ == "__main__":
    # 主测试
    test_improved_cos()
    
    # 不同问题测试
    test_with_different_questions()
    
    print("\n🎉 测试完成！")
