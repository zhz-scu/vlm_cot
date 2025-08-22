#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain-of-Spot 演示脚本

演示Chain-of-Spot方法与其他CoT方法的对比效果
"""

import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 添加src路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chain_of_spot import ChainOfSpotModel, BoundingBox


def create_demo_image() -> str:
    """创建演示图像 - 包含多个对象的复杂场景"""
    # 创建一个更复杂的测试图像
    image = Image.new('RGB', (300, 200), color='lightblue')
    draw = ImageDraw.Draw(image)
    
    # 绘制多个几何图形
    draw.rectangle([20, 20, 80, 80], fill='red', outline='black', width=2)
    draw.circle([150, 50], 30, fill='green', outline='black', width=2)
    draw.polygon([(220, 20), (280, 20), (250, 80)], fill='yellow', outline='black', width=2)
    
    # 添加一些小物件
    draw.rectangle([100, 120, 140, 160], fill='purple', outline='black', width=1)
    draw.circle([200, 140], 15, fill='orange', outline='black', width=1)
    
    # 保存图像
    image_path = "demo_complex_scene.png"
    image.save(image_path)
    return image_path


def simulate_basic_cot(image_path: str, question: str) -> dict:
    """模拟基础CoT方法"""
    print("🔍 基础CoT推理...")
    time.sleep(0.5)  # 模拟推理时间
    
    return {
        "method": "基础CoT",
        "answer": "图像中有多个几何图形：红色正方形、绿色圆形、黄色三角形等。",
        "reasoning": "1. 观察整个图像\n2. 识别所有图形\n3. 描述颜色和形状",
        "roi_focus": False,
        "detail_level": "中等"
    }


def simulate_cos_reasoning(image_path: str, question: str) -> dict:
    """模拟Chain-of-Spot推理"""
    print("🎯 Chain-of-Spot交互式推理...")
    
    # 模拟两步推理过程
    print("  Step 1: 识别关注区域...")
    time.sleep(0.3)
    roi_bbox = BoundingBox(x0=0.4, x1=0.7, y0=0.1, y1=0.6)  # 聚焦绿色圆形区域
    
    print("  Step 2: 基于ROI生成详细答案...")
    time.sleep(0.5)
    
    # 创建ROI可视化
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = roi_bbox.to_coordinates(image.width, image.height)
    draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
    roi_viz_path = "roi_visualization_demo.png"
    image.save(roi_viz_path)
    print(f"  ROI可视化已保存: {roi_viz_path}")
    
    return {
        "method": "Chain-of-Spot",
        "answer": "图像中心区域有一个绿色圆形，直径约60像素，位置在(150,50)附近。该圆形边界清晰，颜色饱和度高，是场景中的显著特征之一。",
        "reasoning": "1. 定位ROI: [0.400,0.700,0.100,0.600]\n2. 聚焦绿色圆形区域\n3. 分析细节特征和属性\n4. 生成精确描述",
        "roi_bbox": roi_bbox.to_string(),
        "roi_focus": True,
        "detail_level": "高",
        "roi_viz_path": roi_viz_path
    }


def compare_methods():
    """对比不同方法的效果"""
    print("🔬 Chain-of-Spot vs 基础CoT 方法对比演示")
    print("=" * 80)
    
    # 创建演示图像
    image_path = create_demo_image()
    print(f"创建演示图像: {image_path}")
    
    question = "请详细描述图像中绿色圆形的特征和位置。"
    print(f"测试问题: {question}")
    print()
    
    # 基础CoT方法
    print("📊 方法1: 基础CoT")
    print("-" * 40)
    basic_result = simulate_basic_cot(image_path, question)
    print(f"推理过程: {basic_result['reasoning']}")
    print(f"答案: {basic_result['answer']}")
    print(f"ROI聚焦: {basic_result['roi_focus']}")
    print(f"细节水平: {basic_result['detail_level']}")
    print()
    
    # Chain-of-Spot方法
    print("🎯 方法2: Chain-of-Spot")
    print("-" * 40)
    cos_result = simulate_cos_reasoning(image_path, question)
    print(f"推理过程: {cos_result['reasoning']}")
    print(f"答案: {cos_result['answer']}")
    print(f"ROI区域: {cos_result['roi_bbox']}")
    print(f"ROI聚焦: {cos_result['roi_focus']}")
    print(f"细节水平: {cos_result['detail_level']}")
    print()
    
    # 对比分析
    print("📈 对比分析")
    print("=" * 80)
    
    comparison = [
        ["方法", "ROI聚焦", "细节水平", "答案长度", "位置信息"],
        ["基础CoT", "否", "中等", len(basic_result['answer']), "无"],
        ["Chain-of-Spot", "是", "高", len(cos_result['answer']), "精确"],
    ]
    
    for row in comparison:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")
    
    print("\n🎯 Chain-of-Spot 优势:")
    print("1. 🔍 动态ROI识别: 自动聚焦问题相关的图像区域")
    print("2. 📊 交互式推理: 两步推理过程提供更精确的分析")
    print("3. 🎨 多粒度特征: 保持原图分辨率同时获取局部细节")
    print("4. 📈 性能提升: 在多个多模态基准上达到SOTA结果")
    print("5. ⚡ 高效计算: 无需增加图像分辨率即可获得细节信息")
    
    print("\n🔬 技术创新:")
    print("- 📍 关联度传播: 基于注意力机制识别ROI")
    print("- 🔄 交互式生成: 对话式推理过程")
    print("- 🎯 自适应聚焦: 根据问题动态调整关注区域")
    print("- 🌟 无需重训练: 可直接应用于现有VLM")
    
    print(f"\n✅ 演示完成! 查看生成的文件:")
    print(f"  - 原始图像: {image_path}")
    print(f"  - ROI可视化: {cos_result.get('roi_viz_path', 'N/A')}")


def usage_example():
    """使用示例"""
    print("\n" + "=" * 80)
    print("📚 Chain-of-Spot 使用示例")
    print("=" * 80)
    
    print("\n1. 命令行使用:")
    print("```bash")
    print("python src/chain_of_spot/cos_inference.py \\")
    print("  --image demo_complex_scene.png \\")
    print("  --question '请描述绿色圆形的位置和特征' \\")
    print("  --device mps --dtype fp16 \\")
    print("  --save-roi-viz")
    print("```")
    
    print("\n2. Python API使用:")
    print("```python")
    print("from src.chain_of_spot import ChainOfSpotModel, cos_generate")
    print("from PIL import Image")
    print("")
    print("# 方法1: 使用高级API")
    print("result = cos_generate(")
    print("    model_id='Qwen/Qwen2.5-VL-3B-Instruct',")
    print("    image_path='demo_complex_scene.png',")
    print("    question='请描述绿色圆形的位置和特征',")
    print("    device='mps'")
    print(")")
    print("")
    print("# 方法2: 使用底层API")
    print("model, processor = load_model(...)")
    print("cos_model = ChainOfSpotModel(model, processor)")
    print("image = Image.open('demo_complex_scene.png')")
    print("response = cos_model.interactive_reasoning(image, question)")
    print("```")
    
    print("\n3. 批量处理:")
    print("```python")
    print("images = [Image.open(f'image_{i}.png') for i in range(5)]")
    print("questions = ['描述主要对象'] * 5")
    print("results = cos_model.batch_reasoning(images, questions)")
    print("```")


if __name__ == "__main__":
    try:
        compare_methods()
        usage_example()
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
