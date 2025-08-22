#!/usr/bin/env python3
import sys
import torch
import argparse
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_NATIVE_QWEN25_VL = True
except ImportError:
    from transformers import AutoModelForCausalLM
    Qwen2_5_VLForConditionalGeneration = None
    HAS_NATIVE_QWEN25_VL = False
from dataclasses import dataclass
from typing import List, Optional

# Model ID - 使用Hugging Face模型名称，会自动使用本地缓存
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

@dataclass
class VoTStep:
    reasoning: str
    visualization: str

def auto_select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def auto_select_dtype(device: str, dtype_arg: str):
    if dtype_arg != "auto":
        mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        return mapping.get(dtype_arg, torch.float32)
    if device == "cuda":
        return torch.bfloat16
    elif device == "mps":
        return torch.float16
    return torch.float32

def load_model_and_processor(model_id: str, torch_dtype, device: str):
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

def build_vot_messages(task_type: str, input_data: str, question: str, images: Optional[List[str]] = None) -> List[dict]:
    # 仿照infer_cot.py的消息构建方式
    user_content = []
    
    if images:
        for img in images:
            user_content.append({"type": "image", "image": img})
    
    # 添加VoT提示
    vot_prompt = (
        f"任务类型: {task_type}\n"
        f"输入数据: {input_data}\n"
        f"问题: {question}\n\n"
        "请使用Visualization-of-Thought (VoT)方法：\n"
        "1. 在每个推理步骤后，用文本形式可视化当前状态（如2D网格、符号表示等）\n"
        "2. 生成交错的推理轨迹和可视化\n"
        "3. 格式：推理步骤：[你的推理] 可视化：[文本形式的可视化]"
    )
    
    user_content.append({"type": "text", "text": vot_prompt})
    
    return [{"role": "user", "content": user_content}]

def generate_vot(
    model,
    processor,
    messages: List[dict],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_steps: int = 3  # 减少步骤数，避免过长
) -> List[VoTStep]:
    results = []
    current_messages = messages.copy()
    
    for step in range(num_steps):
        try:
            # 仿照infer_cot.py的聊天模板应用
            chat_text = processor.apply_chat_template(
                messages=current_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"聊天模板应用失败: {e}", file=sys.stderr)
            # 尝试使用conversation参数
            try:
                chat_text = processor.apply_chat_template(
                    conversation=current_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                print("聊天模板应用成功（使用conversation参数）", file=sys.stderr)
            except Exception as e2:
                print(f"聊天模板应用失败（conversation参数）: {e2}", file=sys.stderr)
                raise
        
        try:
            # 处理视觉信息
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(current_messages)
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
            # 准备输入
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
        
        # 移动到设备
        if device in ("cuda", "mps"):
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        # 生成
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
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        try:
            generated_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except Exception as e:
            print(f"解码失败: {e}", file=sys.stderr)
            raise
        
        # 解析推理和可视化
        if "推理步骤：" in generated_text and "可视化：" in generated_text:
            parts = generated_text.split("可视化：")
            if len(parts) == 2:
                reasoning_part = parts[0].split("推理步骤：")[-1].strip()
                visualization = parts[1].strip()
                results.append(VoTStep(reasoning=reasoning_part, visualization=visualization))
                
                # 添加到消息历史
                current_messages.append({"role": "assistant", "content": generated_text})
                current_messages.append({"role": "user", "content": "继续下一步推理，使用VoT方法。"})
            else:
                break
        else:
            # 如果没有找到特定格式，直接使用整个文本
            results.append(VoTStep(reasoning=generated_text, visualization=""))
            break
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Implement Visualization-of-Thought (VoT) prompting for spatial reasoning.")
    parser.add_argument("--task_type", type=str, required=True, choices=["nl_navigation", "visual_navigation", "visual_tiling"], help="Type of spatial task")
    parser.add_argument("--input_data", type=str, required=True, help="Input data (e.g., map description or grid)")
    parser.add_argument("--question", type=str, required=True, help="The question to solve")
    parser.add_argument("--images", nargs="*", default=[], help="Optional image paths or URLs for visual tasks")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type: auto, bf16, fp16, fp32")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate per step")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num_steps", type=int, default=1, help="Number of VoT steps")
    
    args = parser.parse_args()
    
    device = auto_select_device(args.device)
    torch_dtype = auto_select_dtype(device, args.dtype)
    
    print(f"Using device: {device}, dtype: {torch_dtype}", file=sys.stderr)
    
    model, processor = load_model_and_processor(MODEL_ID, torch_dtype, device)
    
    messages = build_vot_messages(args.task_type, args.input_data, args.question, args.images)
    
    results = generate_vot(model, processor, messages, device, args.max_new_tokens, args.temperature, args.top_p, args.num_steps)
    
    print("VoT Reasoning Results:")
    for i, step in enumerate(results, 1):
        print(f"Step {i}:")
        print(f"Reasoning: {step.reasoning}")
        print(f"Visualization:\n{step.visualization}")
        print("-" * 50)

if __name__ == "__main__":
    main()
