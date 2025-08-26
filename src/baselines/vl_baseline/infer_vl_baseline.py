#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import List, Optional

import torch
from PIL import Image

# 允许从项目根目录调用
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    HAS_NATIVE_QWEN25_VL = True
except Exception:
    from transformers import AutoModelForCausalLM, AutoProcessor
    Qwen2_5_VLForConditionalGeneration = None
    HAS_NATIVE_QWEN25_VL = False


def auto_select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def auto_select_dtype(device: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg != "auto":
        mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        return mapping.get(dtype_arg, torch.float32)
    if device in ("npu", "mps", "xpu"):
        return torch.float16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def move_to_device(inputs: dict, device: str) -> dict:
    if device not in ("cuda", "mps", "npu", "xpu"):
        return inputs
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, 'to'):
            try:
                moved[k] = v.to(device)
            except Exception:
                moved[k] = v
        else:
            moved[k] = v
    return moved


def build_messages(images: List[str], question: str) -> List[dict]:
    content = []
    for p in images:
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": question or "请回答图像相关问题。"})
    return [{"role": "user", "content": content}]


def load_model_and_processor(model_id: str, torch_dtype: torch.dtype, device: str):
    if HAS_NATIVE_QWEN25_VL and Qwen2_5_VLForConditionalGeneration is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device in ("cuda", "mps", "npu", "xpu") else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device in ("cuda", "mps", "npu", "xpu") else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # 手动放到NPU
    if device == "npu" and hasattr(torch, 'npu') and torch.npu.is_available():
        model = model.to("npu")
    return model, processor


def vl_baseline_generate(
    model_id: str,
    images: List[str],
    question: str,
    device: str = "auto",
    dtype_str: str = "auto",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    seed: Optional[int] = None,
) -> dict:
    device = auto_select_device(device)
    torch_dtype = auto_select_dtype(device, dtype_str)
    if seed is not None:
        torch.manual_seed(seed)

    model, processor = load_model_and_processor(model_id, torch_dtype, device)

    # 确保图像存在并转换为PIL
    pil_images = []
    for p in images:
        if isinstance(p, Image.Image):
            pil_images.append(p)
        else:
            if not os.path.exists(p):
                raise FileNotFoundError(f"图像不存在: {p}")
            pil_images.append(Image.open(p).convert("RGB"))

    messages = build_messages(images, question)

    # 模板
    try:
        chat_text = processor.apply_chat_template(
            messages=messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        chat_text = processor.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )

    # 处理输入
    inputs = processor(
        text=[chat_text],
        images=pil_images,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_to_device(inputs, device)

    # 生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )

    # 仅解码新增部分
    try:
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    except Exception:
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return {
        "device": device,
        "dtype": str(torch_dtype),
        "answer": text,
        "method": "vl_baseline",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VL Baseline (no CoT/MCOT)")
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image", type=str, action="append", required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    result = vl_baseline_generate(
        model_id=args.model_id,
        images=args.image,
        question=args.question,
        device=args.device,
        dtype_str=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=== VL Baseline 结果 ===")
        print(f"设备: {result['device']}")
        print(f"精度: {result['dtype']}")
        print(f"答案: {result['answer']}")


if __name__ == "__main__":
    main()
