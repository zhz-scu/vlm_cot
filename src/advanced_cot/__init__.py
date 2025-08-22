"""
高级多模态链式思维推理实现

提供高级的CoT推理功能，集成多种高新技术：
- advanced_cot: 高级多模态推理引擎，集成注意力机制和置信度校准
- advanced_mcot: 多模态CoT实现，支持复杂的多步推理
- enhanced_cot: 增强型CoT，基于ScienceQA的优化实现
"""

from .advanced_cot import advanced_generate, MultiModalReasoningEngine
from .advanced_mcot import AdvancedMCOTEngine
from .enhanced_cot import enhanced_mcot_generate

__all__ = [
    "advanced_generate", 
    "MultiModalReasoningEngine",
    "AdvancedMCOTEngine", 
    "enhanced_mcot_generate"
]
