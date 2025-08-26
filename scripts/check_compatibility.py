#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本兼容性检查脚本

检查PyTorch、Transformers等库的版本兼容性
"""

import sys
import subprocess
import pkg_resources

def check_package_version(package_name):
    """检查包版本"""
    try:
        version = pkg_resources.get_distribution(package_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return "未安装"

def check_torch_compatibility():
    """检查PyTorch兼容性"""
    print("=== PyTorch兼容性检查 ===")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 检查FILE_LIKE是否可用
        try:
            from torch.serialization import FILE_LIKE
            print("✅ FILE_LIKE 可用")
        except ImportError:
            print("❌ FILE_LIKE 不可用 (这是正常的，新版本已移除)")
        
        # 检查其他关键API
        try:
            from torch.serialization import _get_safe_weights_metadata
            print("✅ _get_safe_weights_metadata 可用")
        except ImportError:
            print("❌ _get_safe_weights_metadata 不可用")
        
        # 检查设备支持
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"MPS可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
        print(f"NPU可用: {hasattr(torch, 'npu') and torch.npu.is_available()}")
        print(f"XPU可用: {hasattr(torch, 'xpu') and torch.xpu.is_available()}")
        
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    return True

def check_transformers_compatibility():
    """检查Transformers兼容性"""
    print("\n=== Transformers兼容性检查 ===")
    
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
        
        # 检查关键类是否可用
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            print("✅ AutoModelForCausalLM 可用")
            print("✅ AutoProcessor 可用")
        except ImportError as e:
            print(f"❌ AutoModelForCausalLM/AutoProcessor 不可用: {e}")
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            print("✅ Qwen2_5_VLForConditionalGeneration 可用")
        except ImportError:
            print("❌ Qwen2_5_VLForConditionalGeneration 不可用 (可能需要安装qwen-vl)")
        
    except ImportError as e:
        print(f"❌ Transformers导入失败: {e}")
        return False
    
    return True

def check_npu_compatibility():
    """检查NPU兼容性"""
    print("\n=== NPU兼容性检查 ===")
    
    try:
        import torch_npu
        print("✅ torch_npu 已安装")
        
        try:
            import torch_npu.npu
            print("✅ torch_npu.npu 可用")
        except ImportError as e:
            print(f"❌ torch_npu.npu 不可用: {e}")
        
    except ImportError:
        print("❌ torch_npu 未安装")
        print("   NPU功能将不可用，但不影响其他设备")
        return False
    
    return True

def check_other_dependencies():
    """检查其他依赖"""
    print("\n=== 其他依赖检查 ===")
    
    dependencies = [
        "PIL",
        "numpy",
        "requests",
        "safetensors",
        "accelerate",
        "torchvision"
    ]
    
    for dep in dependencies:
        try:
            if dep == "PIL":
                import PIL
                version = PIL.__version__
            elif dep == "torchvision":
                import torchvision
                version = torchvision.__version__
                # 检查NMS可用性
                if hasattr(torchvision.ops, 'nms'):
                    print(f"✅ {dep}: {version} (NMS可用)")
                else:
                    print(f"⚠️  {dep}: {version} (NMS不可用)")
                continue
            else:
                version = pkg_resources.get_distribution(dep).version
            print(f"✅ {dep}: {version}")
        except (ImportError, pkg_resources.DistributionNotFound):
            print(f"❌ {dep}: 未安装")

def suggest_fixes():
    """建议修复方案"""
    print("\n=== 修复建议 ===")
    
    print("如果遇到FILE_LIKE错误，请尝试以下解决方案:")
    print()
    print("1. 更新transformers库:")
    print("   pip install --upgrade transformers")
    print()
    print("2. 如果问题持续，尝试降级PyTorch:")
    print("   pip install torch==2.0.1")
    print()
    print("3. 对于NPU环境，确保使用兼容版本:")
    print("   pip install torch_npu")
    print()
    print("4. 清理缓存并重新安装:")
    print("   pip cache purge")
    print("   pip install --force-reinstall transformers")

def main():
    """主函数"""
    print("🧠 版本兼容性检查")
    print("=" * 50)
    
    # 检查各个组件
    torch_ok = check_torch_compatibility()
    transformers_ok = check_transformers_compatibility()
    npu_ok = check_npu_compatibility()
    check_other_dependencies()
    
    # 总结
    print("\n=== 检查总结 ===")
    if torch_ok and transformers_ok:
        print("✅ 基本兼容性检查通过")
        if npu_ok:
            print("✅ NPU支持可用")
        else:
            print("⚠️  NPU支持不可用，但不影响其他功能")
    else:
        print("❌ 存在兼容性问题")
        suggest_fixes()

if __name__ == "__main__":
    main()
