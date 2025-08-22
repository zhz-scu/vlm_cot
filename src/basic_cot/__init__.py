"""
基础链式思维推理实现

提供基础的CoT推理功能：
- infer_cot: 基础的多模态CoT推理脚本，支持多种输出风格
"""

from .infer_cot import generate, parse_args, main

__all__ = ["generate", "parse_args", "main"]
