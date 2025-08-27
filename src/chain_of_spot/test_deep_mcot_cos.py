#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from deep_mcot_cos import deep_mcot_cos_generate


def test_deep_mcot_cos():
    """测试深度MCOT+CoS方法"""
    
    print("🧠 测试深度MCOT+Chain-of-Spot方法")
    print("=" * 60)
    
    # 测试配置
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_path = "test_simple.png"
    question = "图片中有什么？请详细分析。"
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return False
    
    try:
        # 执行深度推理
        result = deep_mcot_cos_generate(
            model_id=model_id,
            image_path=image_path,
            question=question,
            device="mps",
            detector_type="yolo",
            save_visualization=True
        )
        
        print("\n" + "=" * 60)
        print("📊 深度MCOT+CoS测试结果")
        print("=" * 60)
        print(f"方法: {result['method']}")
        print(f"设备: {result['device']}")
        print(f"检测器: {result['detector_type']}")
        print(f"问题: {result['question']}")
        print(f"检测区域数: {len(result['detected_regions'])}")
        
        print(f"\n⏱️ 时间统计:")
        print(f"  检测耗时: {result['timing']['detection_time']:.2f}秒")
        print(f"  标注耗时: {result['timing']['annotation_time']:.2f}秒")
        print(f"  CoT推理耗时: {result['timing']['cot_time']:.2f}秒")
        print(f"  总耗时: {result['timing']['total_time']:.2f}秒")
        print(f"  置信度: {result['confidence']:.3f}")
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_methods():
    """对比不同方法"""
    
    print("\n🔍 方法对比测试")
    print("=" * 60)
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_path = "test_simple.png"
    question = "图片中有什么？"
    
    methods = [
        ("edge", "传统边缘检测"),
        ("yolo", "YOLOv8检测")
    ]
    
    results = {}
    
    for detector_type, method_name in methods:
        print(f"\n📊 测试 {method_name}...")
        try:
            result = deep_mcot_cos_generate(
                model_id=model_id,
                image_path=image_path,
                question=question,
                device="mps",
                detector_type=detector_type,
                save_visualization=False
            )
            
            results[detector_type] = {
                "name": method_name,
                "regions": len(result['detected_regions']),
                "total_time": result['timing']['total_time'],
                "detection_time": result['timing']['detection_time'],
                "annotation_time": result['timing']['annotation_time'],
                "cot_time": result['timing']['cot_time'],
                "confidence": result['confidence']
            }
            
            print(f"  ✅ {method_name} 完成")
            
        except Exception as e:
            print(f"  ❌ {method_name} 失败: {e}")
            results[detector_type] = {"name": method_name, "error": str(e)}
    
    # 输出对比结果
    print("\n" + "=" * 60)
    print("📈 方法对比结果")
    print("=" * 60)
    
    for detector_type, result in results.items():
        if "error" in result:
            print(f"{result['name']}: ❌ {result['error']}")
        else:
            print(f"{result['name']}:")
            print(f"  检测区域数: {result['regions']}")
            print(f"  检测耗时: {result['detection_time']:.2f}秒")
            print(f"  标注耗时: {result['annotation_time']:.2f}秒")
            print(f"  CoT推理耗时: {result['cot_time']:.2f}秒")
            print(f"  总耗时: {result['total_time']:.2f}秒")
            print(f"  置信度: {result['confidence']:.3f}")


if __name__ == "__main__":
    # 主测试
    success = test_deep_mcot_cos()
    
    if success:
        # 方法对比
        compare_methods()
        
        print("\n🎉 深度MCOT+CoS测试完成！")
        print("💾 生成的文件:")
        files = [
            "/test_res/deep_mcot_detection_yolo.png",
            "deep_mcot_cos_visualization.png"
        ]
        
        for file in files:
            if os.path.exists(file):
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (未生成)")
    else:
        print("\n❌ 测试失败")
