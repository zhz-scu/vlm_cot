#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import os
from typing import List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_NATIVE_QWEN25_VL = True
except ImportError:
    from transformers import AutoModelForCausalLM
    Qwen2_5_VLForConditionalGeneration = None
    HAS_NATIVE_QWEN25_VL = False

# 直接导入工具函数
try:
    from src.core.qwen_vl_utils import process_vision_info
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from core.qwen_vl_utils import process_vision_info
    except ImportError:
        print("警告: 无法导入qwen_vl_utils，使用简化版本")
        def process_vision_info(messages):
            image_inputs = []
            video_inputs = []
            for msg in messages:
                for content in msg.get("content", []):
                    if content.get("type") == "image":
                        image_inputs.append(content["image"])
                    elif content.get("type") == "video":
                        video_inputs.append(content["video"])
            return image_inputs, video_inputs


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
            # 测试 MPS 是否真的可用
            test_tensor = torch.tensor([1.0], device="mps")
            del test_tensor
            return "mps"
        except Exception:
            print("警告: MPS 检测到但不可用，回退到 CPU", file=sys.stderr)
            return "cpu"
    
    return "cpu"


def auto_select_dtype(device: str, dtype_arg: str):
    """自动选择最佳数据类型 - 支持NPU"""
    if dtype_arg != "auto":
        mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        return mapping[dtype_arg]
    
    if device == "npu":
        return torch.float16  # NPU推荐使用FP16
    elif device == "cuda":
        return torch.bfloat16
    elif device == "mps":
        # MPS 推荐使用 float16 以获得最佳性能
        return torch.float16
    elif device == "xpu":
        return torch.float16
    return torch.float32


def build_messages(image_inputs: List[str], question: str, cot_style: str) -> List[dict]:
    """构建多模态消息"""
    user_content = []
    for img in image_inputs:
        user_content.append({"type": "image", "image": img})
    
    prompt = question
    if cot_style == "rationale_and_answer":
        prompt = (
            "你是一位严谨的视觉推理助手。请先进行逐步推理，然后在最后用'最终答案：'给出简洁答案。\n"
            "请严格按以下格式输出：\n"
            "思考过程：<你的详细推理步骤>\n"
            "最终答案：<一句话答案>"
        ) + ("\n问题：" + question if question else "")
    elif cot_style == "short_answer":
        prompt = (
            "请先在心中逐步思考，再只输出最终答案的一句话。不要暴露中间推理过程。\n"
        ) + ("\n问题：" + question if question else "")
    else:
        prompt = question or "请逐步思考并回答问题。"

    user_content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]
    return messages


def move_to_device(inputs: dict, device: str) -> dict:
    """将输入张量移动到指定设备 - 支持NPU"""
    if device not in ("cuda", "mps", "npu", "xpu"):
        return inputs
    
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            try:
                moved[k] = v.to(device)
            except Exception as e:
                print(f"警告: 无法移动张量 {k} 到设备 {device}: {e}", file=sys.stderr)
                moved[k] = v
        else:
            moved[k] = v
    return moved


def infer_cot_generate(
    model_id: str,
    images: List[str],
    question: str,
    device: str = "auto",
    dtype_str: str = "auto",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    cot_style: str = "rationale_and_answer",
    seed: Optional[int] = None,
) -> dict:
    """基础CoT生成函数 - 支持NPU"""
    
    # 设备选择
    device = auto_select_device(device)
    torch_dtype = auto_select_dtype(device, dtype_str)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # 加载模型
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        # NPU特定的加载配置
        if device == "npu":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch_dtype,
                device_map=None  # NPU需要手动管理设备映射
            )
            # 手动移动到NPU
            model = model.to("npu")
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch_dtype, device_map="auto"
            )
        
        processor = AutoProcessor.from_pretrained(model_id)
    except:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        if device == "npu":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch_dtype,
                device_map=None,
                trust_remote_code=True
            )
            model = model.to("npu")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
            )
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # 构建消息
    messages = build_messages(images, question, cot_style)
    
    # 应用聊天模板
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 直接加载图像，避免复杂的处理逻辑
    from PIL import Image
    pil_images = []
    for img_path in images:
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                print(f"成功加载图像: {img_path}, 尺寸: {img.size}")
                pil_images.append(img)
            except Exception as e:
                print(f"图像加载失败 {img_path}: {e}")
                raise
        else:
            print(f"错误: 图像文件不存在: {img_path}")
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
    
    # 准备输入
    try:
        print(f"准备输入，文本长度: {len(chat_text)}, 图像数量: {len(pil_images)}")
        inputs = processor(
            text=[chat_text],
            images=pil_images,
            videos=[],
            padding=True,
            return_tensors="pt",
        )
        print(f"输入准备完成，keys: {list(inputs.keys())}")
    except Exception as e:
        print(f"输入准备失败: {e}")
        raise
    
    # 移动到设备
    inputs = move_to_device(inputs, device)
    
    # 生成
    try:
        print(f"开始生成，输入形状: {inputs['input_ids'].shape}")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"生成完成，输出形状: {generated_ids.shape}")
    except Exception as e:
        print(f"生成失败: {e}")
        raise
    
    # 解码 - 修复索引越界问题
    try:
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    except IndexError as e:
        print(f"解码错误: {e}")
        print(f"inputs['input_ids'] 长度: {len(inputs['input_ids'])}")
        print(f"generated_ids 长度: {len(generated_ids)}")
        # 尝试直接解码整个序列
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # 移除输入部分
        if chat_text in output_text:
            output_text = output_text.replace(chat_text, "").strip()
    
    return {
        "answer": output_text,
        "device": device,
        "method": "basic_cot",
        "cot_style": cot_style
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基础CoT推理 - 支持NPU")
    parser.add_argument("--model-id", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", type=str, action="append", required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--cot-style", type=str, default="rationale_and_answer",
                       choices=["rationale_and_answer", "short_answer", "standard"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        result = infer_cot_generate(
            model_id=args.model_id,
            images=args.image,
            question=args.question,
            device=args.device,
            dtype_str=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            cot_style=args.cot_style,
            seed=args.seed,
        )
        
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("=== 基础CoT推理结果 ===")
            print(f"设备: {result['device']}")
            print(f"方法: {result['method']}")
            print(f"CoT风格: {result['cot_style']}")
            print(f"答案: {result['answer']}")
            
    except Exception as e:
        print(f"推理失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
