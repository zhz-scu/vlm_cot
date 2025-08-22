"""
Chain-of-Spot (CoS) - Interactive Reasoning for Large Vision-Language Models

基于论文: "Chain-of-Spot: Interactive Reasoning Improves Large Vision-Language Models"

该方法通过两步交互式推理过程提升视觉语言模型的推理能力：
1. 动态识别图像中的关注区域(ROI)
2. 基于ROI和原始图像生成更准确的答案

核心优势:
- 🎯 交互式推理: 自动聚焦关键区域
- 🔍 多粒度特征: 保持原图分辨率同时获取细节
- 📊 性能提升: 在多个多模态基准上达到SOTA
- ⚡ 高效推理: 无需修改原始图像分辨率
"""

from .cos_model import (
    ChainOfSpotModel,
    BoundingBox,
    CoSResponse,
    AttentionAnalyzer,
    ImageCropper
)

from .cos_inference import cos_generate

__version__ = "1.0.0"
__author__ = "VLM CoT Team"

__all__ = [
    "ChainOfSpotModel",
    "BoundingBox", 
    "CoSResponse",
    "AttentionAnalyzer",
    "ImageCropper",
    "cos_generate"
]
