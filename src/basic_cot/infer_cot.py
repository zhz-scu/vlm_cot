#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from typing import List, Optional

import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_NATIVE_QWEN25_VL = True
except ImportError:
    from transformers import AutoModelForCausalLM
    Qwen2_5_VLForConditionalGeneration = None
    HAS_NATIVE_QWEN25_VL = False
# 修复导入问题
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.core.qwen_vl_utils import process_vision_info
except ImportError:
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

try:
    from src.core.npu_utils import auto_select_device, auto_select_dtype, move_to_device
except ImportError:
    try:
        from core.npu_utils import auto_select_device, auto_select_dtype, move_to_device
    except ImportError:
        print("警告: 无法导入npu_utils，使用内置版本")
        # 使用内置的设备选择函数


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
        # NPU 推荐使用 float16 以获得最佳性能
        return torch.float16
    elif device == "cuda":
        return torch.bfloat16
    elif device == "mps":
        # MPS 推荐使用 float16 以获得最佳性能
        return torch.float16
    elif device == "xpu":
        # XPU 推荐使用 float16
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
                print(f"警告: 无法将 {k} 移动到 {device}: {e}", file=sys.stderr)
                moved[k] = v
        else:
            moved[k] = v
    return moved


def load_model_and_processor(model_id: str, torch_dtype, device: str):
    """加载模型和处理器，支持 MPS"""
    try:
        if HAS_NATIVE_QWEN25_VL and Qwen2_5_VLForConditionalGeneration is not None:
            # 使用原生支持
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device in ("cuda", "mps") else None,
            )
            processor = AutoProcessor.from_pretrained(model_id)
        else:
            # 回退到远程代码实现
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device in ("cuda", "mps") else None,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        return model, processor
    except Exception as e:
        print(f"模型加载失败: {e}", file=sys.stderr)
        raise


def generate(
    model_id: str,
    images: List[str],
    question: str,
    device: str,
    dtype_str: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    cot_style: str,
    seed: Optional[int],
) -> str:
    """生成推理结果"""
    device = auto_select_device(device)
    torch_dtype = auto_select_dtype(device, dtype_str)

    print(f"使用设备: {device}, 数据类型: {torch_dtype}", file=sys.stderr)

    if seed is not None:
        torch.manual_seed(seed)

    try:
        model, processor = load_model_and_processor(model_id, torch_dtype, device)
        print("模型加载成功", file=sys.stderr)
    except Exception as e:
        print(f"模型加载失败: {e}", file=sys.stderr)
        raise

    messages = build_messages(images, question, cot_style)

    try:
        chat_text = processor.apply_chat_template(
            messages=messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("聊天模板应用成功", file=sys.stderr)
    except Exception as e:
        print(f"聊天模板应用失败: {e}", file=sys.stderr)
        # 尝试使用conversation参数
        try:
            chat_text = processor.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            print("聊天模板应用成功（使用conversation参数）", file=sys.stderr)
        except Exception as e2:
            print(f"聊天模板应用失败（conversation参数）: {e2}", file=sys.stderr)
            raise

    try:
        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs is None:
            image_inputs = []
        if video_inputs is None:
            video_inputs = []
        print(f"视觉处理完成: {len(image_inputs)} 张图片, {len(video_inputs)} 个视频", file=sys.stderr)
    except Exception as e:
        print(f"视觉处理失败: {e}", file=sys.stderr)
        image_inputs = []
        video_inputs = []

    try:
        # 只有当有图像时才传递图像参数
        processor_kwargs = {
            "text": [chat_text],
            "padding": True,
            "return_tensors": "pt",
        }
        
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
            
        inputs = processor(**processor_kwargs)
        print("输入处理完成", file=sys.stderr)
    except Exception as e:
        print(f"输入处理失败: {e}", file=sys.stderr)
        raise

    inputs = move_to_device(inputs, device)

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )

    try:
        print("开始生成...", file=sys.stderr)
        generated_ids = model.generate(**inputs, **generation_kwargs)
        print("生成完成", file=sys.stderr)
    except Exception as e:
        print(f"生成失败: {e}", file=sys.stderr)
        raise

    # 仅解码新生成的部分
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    try:
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts[0]
    except Exception as e:
        print(f"解码失败: {e}", file=sys.stderr)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-3B-Instruct 多模态 CoT 推理脚本 (Mac MPS 优化版)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="模型标识（Hugging Face Hub 或本地路径）",
    )
    parser.add_argument(
        "--image",
        type=str,
        action="append",
        required=True,
        help="图片路径或URL。可重复传入多次以提供多张图片。",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="请逐步思考并回答问题。",
        help="要询问的问题。",
    )
    parser.add_argument(
        "--cot-style",
        type=str,
        choices=["rationale_and_answer", "short_answer", "free"],
        default="rationale_and_answer",
        help="CoT 输出风格：rationale_and_answer（思考过程+最终答案）、short_answer（隐藏推理，仅答案）、free（不加约束）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="推理设备选择。auto 会自动选择最佳设备（Mac 上优先 MPS）。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="权重量化精度（自动/手动）。MPS 推荐 auto 或 fp16。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="生成最大新token数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="采样温度（0为贪心）。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="核采样阈值。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选）。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以JSON输出结果，便于程序化处理。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        output_text = generate(
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
    except Exception as exc:  # noqa: BLE001
        print(f"推理失败：{exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps({
            "model": args.model_id,
            "question": args.question,
            "images": args.image,
            "cot_style": args.cot_style,
            "output": output_text,
        }, ensure_ascii=False))
    else:
        print(output_text)


if __name__ == "__main__":
    main()
