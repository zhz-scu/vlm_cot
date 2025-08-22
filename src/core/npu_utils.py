#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU工具类 - 为所有MCOT方法提供NPU支持
"""

import torch
import os
import sys
from typing import Optional, Dict, Any
import warnings


class NPUManager:
    """NPU管理器"""
    
    def __init__(self):
        self.has_npu = self._check_npu_availability()
        self.npu_initialized = False
        
    def _check_npu_availability(self) -> bool:
        """检查NPU可用性"""
        try:
            import torch_npu
            return hasattr(torch, 'npu') and torch.npu.is_available()
        except ImportError:
            return False
    
    def setup_npu_environment(self, device_id: int = 0) -> bool:
        """设置NPU环境"""
        if not self.has_npu:
            warnings.warn("NPU不可用，请安装torch_npu")
            return False
        
        try:
            import torch_npu
            os.environ['ASCEND_DEVICE_ID'] = str(device_id)
            os.environ['ASCEND_VISIBLE_DEVICES'] = str(device_id)
            torch.npu.set_device(device_id)
            self.npu_initialized = True
            print(f"NPU环境初始化完成，设备ID: {device_id}")
            return True
        except Exception as e:
            warnings.warn(f"NPU环境初始化失败: {e}")
            return False
    
    def optimize_model_for_npu(self, model: torch.nn.Module, device: str = "npu") -> torch.nn.Module:
        """为NPU优化模型"""
        if device != "npu" or not self.has_npu:
            return model
        
        try:
            import torch_npu
            model = model.to(device)
            if hasattr(model, 'half'):
                model = model.half()
            try:
                model = torch.npu.optimize(model)
                print("NPU图优化已启用")
            except:
                print("NPU图优化不可用，使用标准模式")
            return model
        except Exception as e:
            warnings.warn(f"NPU模型优化失败: {e}")
            return model
    
    def clear_npu_cache(self):
        """清理NPU缓存"""
        if self.has_npu:
            try:
                import torch_npu
                torch.npu.empty_cache()
            except:
                pass


def auto_select_device(device_arg: str) -> str:
    """自动选择最佳可用设备 - 支持NPU"""
    if device_arg != "auto":
        return device_arg
    
    # 检查 NPU (华为昇腾)
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return "npu"
    
    # 检查 CUDA
    if torch.cuda.is_available():
        return "cuda"
    
    # 检查 XPU (Intel GPU)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    
    # 检查 MPS (Mac)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device="mps")
            del test_tensor
            return "mps"
        except Exception:
            print("警告: MPS 检测到但不可用，回退到 CPU", file=sys.stderr)
            return "cpu"
    
    return "cpu"


def auto_select_dtype(device: str, dtype_arg: str) -> torch.dtype:
    """自动选择最佳数据类型 - 支持NPU"""
    if dtype_arg != "auto":
        mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        return mapping.get(dtype_arg, torch.float32)
    
    if device == "npu":
        return torch.float16  # NPU推荐使用FP16
    elif device == "cuda":
        return torch.bfloat16
    elif device == "mps":
        return torch.float16
    elif device == "xpu":
        return torch.float16
    else:
        return torch.float32


def move_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    """将输入张量移动到指定设备 - 支持NPU"""
    if device not in ("cuda", "mps", "npu", "xpu"):
        return inputs
    
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            try:
                moved[k] = v.to(device)
            except Exception as e:
                print(f"警告: 无法移动张量 {k} 到设备 {device}: {e}")
                moved[k] = v
        else:
            moved[k] = v
    
    return moved


# 全局NPU管理器实例
npu_manager = NPUManager()


def get_npu_info() -> Dict[str, Any]:
    """获取NPU信息"""
    info = {
        "available": npu_manager.has_npu,
        "initialized": npu_manager.npu_initialized,
        "device_count": 0
    }
    
    if npu_manager.has_npu:
        try:
            import torch_npu
            info["device_count"] = torch.npu.device_count()
        except Exception as e:
            info["error"] = str(e)
    
    return info


def print_npu_status():
    """打印NPU状态"""
    info = get_npu_info()
    print("=== NPU状态 ===")
    print(f"可用: {info['available']}")
    print(f"已初始化: {info['initialized']}")
    print(f"设备数量: {info['device_count']}")
    
    if 'error' in info:
        print(f"错误: {info['error']}")


if __name__ == "__main__":
    print_npu_status()
