#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import sys
import time

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from improved_cos_model import ObjectDetector, DetectedRegion


def create_test_image():
    """创建测试图像"""
    width, height = 400, 300
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 绘制一些简单的几何图形
    draw.ellipse([50, 50, 150, 150], fill='red', outline='darkred', width=3)
    draw.rectangle([200, 50, 350, 150], fill='blue', outline='darkblue', width=3)
    points = [(100, 200), (200, 200), (150, 250)]
    draw.polygon(points, fill='green', outline='darkgreen', width=3)
    draw.rectangle([250, 200, 350, 250], fill='yellow', outline='orange', width=3)
    
    return image


def create_visualization(image, regions):
    """创建可视化结果"""
    viz_image = image.copy()
    draw = ImageDraw.Draw(viz_image)
    
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    
    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = region.bbox
        
        img_width, img_height = image.size
        x0_pixel = int(x0 * img_width)
        y0_pixel = int(y0 * img_height)
        x1_pixel = int(x1 * img_width)
        y1_pixel = int(y1 * img_height)
        
        draw.rectangle([x0_pixel, y0_pixel, x1_pixel, y1_pixel], outline=color, width=3)
        
        label = f"R{i}: {region.confidence:.2f}"
        if hasattr(region, 'label') and region.label:
            label += f" ({region.label})"
        
        text_x = max(0, x0_pixel)
        text_y = max(0, y0_pixel - 20)
        draw.text((text_x, text_y), label, fill=color)
    
    return viz_image


def test_edge_detection():
    """测试边缘检测功能"""
    print("🔍 测试边缘检测功能")
    print("=" * 50)
    
    test_image = create_test_image()
    test_image.save("debug_test_image.png")
    print("✅ 创建测试图像: debug_test_image.png")
    
    detector = ObjectDetector("cpu")
    detector.initialize_detector("edge")
    
    if not detector.initialized:
        print("❌ 边缘检测器初始化失败")
        return False
    
    print("✅ 边缘检测器初始化成功")
    
    start_time = time.time()
    regions = detector.detect_regions(test_image, min_confidence=0.1)
    detection_time = time.time() - start_time
    
    print(f"✅ 检测完成，耗时: {detection_time:.3f}秒")
    print(f"📊 检测到 {len(regions)} 个区域")
    
    for i, region in enumerate(regions):
        print(f"  区域 {i}: bbox={region.bbox}, 置信度={region.confidence:.3f}")
    
    viz_image = create_visualization(test_image, regions)
    viz_image.save("debug_edge_detection_result.png")
    print("✅ 保存可视化结果: debug_edge_detection_result.png")
    
    return len(regions) > 0


def test_opencv_basic():
    """测试OpenCV基本功能"""
    print("\n🔍 测试OpenCV基本功能")
    print("=" * 50)
    
    try:
        print(f"✅ OpenCV版本: {cv2.__version__}")
        
        test_image = create_test_image()
        img_array = np.array(test_image)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        print("✅ 图像转换成功")
        
        edges = cv2.Canny(gray, 50, 150)
        print("✅ Canny边缘检测成功")
        
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"✅ 找到 {len(contours)} 个轮廓")
        
        cv2.imwrite("debug_opencv_edges.png", edges)
        print("✅ 保存边缘检测结果: debug_opencv_edges.png")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🎯 目标检测器调试测试")
    print("=" * 60)
    
    opencv_ok = test_opencv_basic()
    edge_ok = test_edge_detection()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    print(f"OpenCV基本功能: {'✅ 通过' if opencv_ok else '❌ 失败'}")
    print(f"边缘检测: {'✅ 通过' if edge_ok else '❌ 失败'}")
    
    if edge_ok:
        print("\n🎉 目标检测器调试成功！")
    else:
        print("\n⚠️ 目标检测器存在问题，需要进一步调试")


if __name__ == "__main__":
    main()
