"""
VLM CoT (Visual Language Model Chain-of-Thought) 推理系统

这个包提供了多种多模态链式思维推理的实现，包括：
- 基础CoT实现
- 高级CoT实现  
- 专门化CoT实现
- 实验性CoT实现

主要模块：
- basic_cot: 基础链式思维推理
- advanced_cot: 高级多模态推理引擎
- specialized_cot: 专门化推理（ScienceQA、分层推理等）
- experimental: 实验性推理方法（VoT等）
- core: 核心工具和组件
"""

__version__ = "1.0.0"
__author__ = "VLM CoT Team"

# 简化导入，避免复杂的模块导入
__all__ = [
    "basic_cot",
    "advanced_cot", 
    "chain_of_spot",
    "experimental",
    "core"
]
