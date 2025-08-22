"""
实验性链式思维推理实现

提供实验性的CoT推理方法：
- vot_prompting: Visualization-of-Thought (VoT) 提示方法，用于空间推理任务
"""

from .vot_prompting import generate_vot, VoTStep, main

__all__ = ["generate_vot", "VoTStep", "main"]
