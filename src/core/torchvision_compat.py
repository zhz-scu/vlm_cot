#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torchvision兼容性修复模块

解决torchvision::NMS不存在的问题
"""

import torch
import warnings
from typing import Optional, Tuple

def check_torchvision_nms():
    """检查torchvision NMS是否可用"""
    try:
        import torchvision
        # 尝试导入NMS
        if hasattr(torchvision.ops, 'nms'):
            return True
        else:
            return False
    except ImportError:
        return False

def custom_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """自定义NMS实现，当torchvision NMS不可用时使用"""
    
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # 按分数排序
    _, order = scores.sort(0, descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i)
        
        # 计算IoU
        xx1 = boxes[order[1:], 0].clamp(min=boxes[i, 0])
        yy1 = boxes[order[1:], 1].clamp(min=boxes[i, 1])
        xx2 = boxes[order[1:], 2].clamp(max=boxes[i, 2])
        yy2 = boxes[order[1:], 3].clamp(max=boxes[i, 3])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        ovr = inter / (boxes[i, 2] * boxes[i, 3] + boxes[order[1:], 2] * boxes[order[1:], 3] - inter)
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def safe_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """安全的NMS调用，自动选择可用的实现"""
    
    if check_torchvision_nms():
        # 使用torchvision的NMS
        try:
            import torchvision.ops
            return torchvision.ops.nms(boxes, scores, iou_threshold)
        except Exception as e:
            warnings.warn(f"torchvision NMS失败，使用自定义实现: {e}")
            return custom_nms(boxes, scores, iou_threshold)
    else:
        # 使用自定义NMS
        warnings.warn("torchvision NMS不可用，使用自定义实现")
        return custom_nms(boxes, scores, iou_threshold)

def fix_torchvision_imports():
    """修复torchvision导入问题"""
    
    # 检查并修复常见的torchvision问题
    try:
        import torchvision
        print(f"torchvision版本: {torchvision.__version__}")
        
        # 检查关键功能
        if hasattr(torchvision.ops, 'nms'):
            print("✅ torchvision NMS 可用")
        else:
            print("❌ torchvision NMS 不可用")
            
        if hasattr(torchvision.transforms, 'functional'):
            print("✅ torchvision transforms 可用")
        else:
            print("❌ torchvision transforms 不可用")
            
    except ImportError as e:
        print(f"❌ torchvision导入失败: {e}")
        return False
    
    return True

def get_recommended_versions():
    """获取推荐的版本组合"""
    return {
        "torch": "2.0.1",
        "torchvision": "0.15.2", 
        "transformers": "4.50.0",
        "说明": "这些版本组合经过测试，兼容性较好"
    }

def install_compatible_versions():
    """安装兼容版本"""
    versions = get_recommended_versions()
    
    print("推荐安装以下版本组合:")
    for package, version in versions.items():
        if package != "说明":
            print(f"  {package}=={version}")
    
    print("\n安装命令:")
    print(f"pip install torch=={versions['torch']} torchvision=={versions['torchvision']}")
    print(f"pip install transformers=={versions['transformers']}")

if __name__ == "__main__":
    print("=== torchvision兼容性检查 ===")
    fix_torchvision_imports()
    print("\n=== 推荐版本 ===")
    install_compatible_versions()
