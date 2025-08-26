#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from improved_cos_model import ObjectDetector, DetectedRegion


def test_detection_only():
    """只测试目标检测功能，不加载大模型"""
    
    print("🔬 测试改进CoS的目标检测功能")
    print("=" * 60)
    
    # 使用测试图像
    image_path = "test_simple.png"
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return False
    
    # 加载图像
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✅ 图像加载成功: {image.size}")
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        return False
    
    # 测试边缘检测器
    print("\n📊 测试边缘检测器...")
    detector = ObjectDetector("cpu")
    detector.initialize_detector("edge")
    
    if not detector.initialized:
        print("❌ 边缘检测器初始化失败")
        return False
    
    print("✅ 边缘检测器初始化成功")
    
    # 执行检测
    start_time = time.time()
    regions = detector.detect_regions(image, min_confidence=0.1)
    detection_time = time.time() - start_time
    
    print(f"✅ 检测完成，耗时: {detection_time:.3f}秒")
    print(f"📊 检测到 {len(regions)} 个区域")
    
    # 显示检测结果
    for i, region in enumerate(regions):
        print(f"  区域 {i}: bbox={region.bbox}, 置信度={region.confidence:.3f}")
    
    # 创建可视化
    from improved_cos_model import _create_visualization
    viz_image = _create_visualization(image, regions)
    viz_image.save("improved_cos_detection_result.png")
    print("✅ 保存可视化结果: improved_cos_detection_result.png")
    
    return len(regions) > 0


def test_with_different_confidence():
    """测试不同置信度阈值的效果"""
    
    print("\n🔍 测试不同置信度阈值")
    print("=" * 60)
    
    image_path = "test_simple.png"
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    detector = ObjectDetector("cpu")
    detector.initialize_detector("edge")
    
    confidence_thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for threshold in confidence_thresholds:
        print(f"\n📊 置信度阈值: {threshold}")
        regions = detector.detect_regions(image, min_confidence=threshold)
        print(f"  检测到 {len(regions)} 个区域")
        
        for i, region in enumerate(regions):
            print(f"    区域 {i}: 置信度={region.confidence:.3f}, bbox={region.bbox}")


def test_region_cropping():
    """测试区域裁剪功能"""
    
    print("\n🔍 测试区域裁剪功能")
    print("=" * 60)
    
    image_path = "test_simple.png"
    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    detector = ObjectDetector("cpu")
    detector.initialize_detector("edge")
    
    regions = detector.detect_regions(image, min_confidence=0.1)
    
    print(f"📊 检测到 {len(regions)} 个区域，测试裁剪...")
    
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region.bbox
        img_width, img_height = image.size
        
        x0_pixel = max(0, min(int(x0 * img_width), img_width))
        y0_pixel = max(0, min(int(y0 * img_height), img_height))
        x1_pixel = max(x0_pixel, min(int(x1 * img_width), img_width))
        y1_pixel = max(y0_pixel, min(int(y1 * img_height), img_height))
        
        if x1_pixel > x0_pixel and y1_pixel > y0_pixel:
            try:
                cropped = image.crop((x0_pixel, y0_pixel, x1_pixel, y1_pixel))
                cropped.save(f"cropped_region_{i}.png")
                print(f"✅ 区域 {i} 裁剪成功: {cropped.size}")
            except Exception as e:
                print(f"❌ 区域 {i} 裁剪失败: {e}")
        else:
            print(f"⚠️ 区域 {i} 坐标无效")


def main():
    """主测试函数"""
    print("🎯 改进CoS目标检测功能测试")
    print("=" * 60)
    
    # 测试基本检测功能
    detection_ok = test_detection_only()
    
    if detection_ok:
        # 测试不同置信度
        test_with_different_confidence()
        
        # 测试区域裁剪
        test_region_cropping()
        
        print("\n" + "=" * 60)
        print("🎉 目标检测功能测试完成！")
        print("💡 现在可以集成到完整的改进CoS方法中")
        
        print("\n📁 生成的文件:")
        files = [
            "improved_cos_detection_result.png",
            "cropped_region_0.png",
            "cropped_region_1.png",
            "cropped_region_2.png",
            "cropped_region_3.png"
        ]
        
        for file in files:
            if os.path.exists(file):
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (未生成)")
    else:
        print("\n❌ 目标检测功能测试失败")


if __name__ == "__main__":
    main()
