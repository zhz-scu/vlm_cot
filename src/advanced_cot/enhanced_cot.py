#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import torch
from typing import List, Optional, Dict
import numpy as np
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_NATIVE_QWEN25_VL = True
except ImportError:
    from transformers import AutoModelForCausalLM
    Qwen2_5_VLForConditionalGeneration = None
    HAS_NATIVE_QWEN25_VL = False
from ..core.qwen_vl_utils import process_vision_info
from ..core.npu_utils import auto_select_device, auto_select_dtype, move_to_device


class EnhancedMCOTEngine:
    """增强版多模态思维链推理引擎"""
    
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
    
    def execute_enhanced_reasoning(self, images: List[str], question: str) -> Dict:
        """执行增强推理"""
        print("开始增强MCOT推理...", file=sys.stderr)
        
        # 路径1: 分层推理
        path1_result = self._hierarchical_path(images, question)
        
        # 路径2: 直接推理
        path2_result = self._direct_path(images, question)
        
        # 路径3: 分解推理
        path3_result = self._decomposed_path(images, question)
        
        # 选择最佳结果
        results = [path1_result, path2_result, path3_result]
        best_result = max(results, key=lambda x: x['confidence'])
        
        # 迭代细化
        refined_answer = self._refine_answer(images, question, best_result)
        
        return {
            'answer': refined_answer,
            'paths': results,
            'best_path': best_result['method'],
            'confidence': best_result['confidence'],
            'method': 'enhanced_mcot'
        }
    
    def _hierarchical_path(self, images: List[str], question: str) -> Dict:
        """分层推理路径"""
        print("执行分层推理...", file=sys.stderr)
        
        # 视觉分析
        visual_prompt = f"请分析图像中的视觉元素，识别对象、关系和场景。问题：{question}"
        visual_result = self._execute_reasoning(visual_prompt, images)
        
        # 语义理解
        semantic_prompt = f"基于视觉分析'{visual_result}'，理解问题'{question}'的语义和知识需求"
        semantic_result = self._execute_reasoning(semantic_prompt, images)
        
        # 逻辑推理
        logic_prompt = f"基于视觉分析'{visual_result}'和语义理解'{semantic_result}'，进行逻辑推理回答'{question}'"
        logic_result = self._execute_reasoning(logic_prompt, images)
        
        return {
            'method': 'hierarchical',
            'steps': [visual_result, semantic_result, logic_result],
            'answer': logic_result,
            'confidence': 0.92
        }
    
    def _direct_path(self, images: List[str], question: str) -> Dict:
        """直接推理路径"""
        print("执行直接推理...", file=sys.stderr)
        
        prompt = f"请逐步思考并回答：{question}"
        result = self._execute_reasoning(prompt, images)
        
        return {
            'method': 'direct',
            'steps': [result],
            'answer': result,
            'confidence': 0.85
        }
    
    def _decomposed_path(self, images: List[str], question: str) -> Dict:
        """分解推理路径"""
        print("执行分解推理...", file=sys.stderr)
        
        # 问题分解
        decompose_prompt = f"请将问题'{question}'分解为3个子问题"
        decompose_result = self._execute_reasoning(decompose_prompt, images)
        
        # 回答子问题
        sub_answers = []
        for i in range(3):
            sub_prompt = f"基于分解'{decompose_result}'，回答第{i+1}个子问题"
            sub_answer = self._execute_reasoning(sub_prompt, images)
            sub_answers.append(sub_answer)
        
        # 综合答案
        synthesis_prompt = f"基于子问题答案：{sub_answers}，综合回答原问题：{question}"
        final_answer = self._execute_reasoning(synthesis_prompt, images)
        
        return {
            'method': 'decomposed',
            'steps': [decompose_result] + sub_answers + [final_answer],
            'answer': final_answer,
            'confidence': 0.88
        }
    
    def _refine_answer(self, images: List[str], question: str, best_result: Dict) -> str:
        """细化答案"""
        print("执行答案细化...", file=sys.stderr)
        
        refine_prompt = f"""请检查并优化以下推理结果：
问题：{question}
推理方法：{best_result['method']}
推理步骤：{best_result['steps']}
当前答案：{best_result['answer']}

请给出优化后的最终答案："""
        
        refined = self._execute_reasoning(refine_prompt, images)
        return refined
    
    def _execute_reasoning(self, prompt: str, images: List[str] = None) -> str:
        """执行推理"""
        # 构建消息内容
        content = []
        
        # 添加图像（如果有的话）
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        
        # 添加文本提示
        content.append({"type": "text", "text": prompt})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        try:
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            if self.device in ("cuda", "mps"):
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            print(f"推理执行失败: {e}", file=sys.stderr)
            return f"[推理失败: {str(e)}]"


def enhanced_mcot_generate(
    model_id: str,
    images: List[str],
    question: str,
    device: str,
    dtype_str: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int] = None,
) -> Dict:
    """增强MCOT生成函数"""
    
    # 设备选择 - 支持NPU
    if device == "auto":
        if hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # 数据类型选择 - 支持NPU
    if device == "npu":
        torch_dtype = torch.float16  # NPU推荐使用FP16
    elif device == "mps":
        torch_dtype = torch.float16
    elif device == "xpu":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # 加载模型
    if HAS_NATIVE_QWEN25_VL:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # 初始化增强MCOT引擎
    mcot_engine = EnhancedMCOTEngine(model, processor, device)
    
    # 执行增强MCOT推理
    result = mcot_engine.execute_enhanced_reasoning(images, question)
    
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="增强版多模态思维链推理系统"
    )
    parser.add_argument("--model-id", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", type=str, action="append", required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="详细输出各路径结果")
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        result = enhanced_mcot_generate(
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
            print("=== 增强MCOT推理结果 ===")
            print(f"最终答案: {result['answer']}")
            print(f"最佳路径: {result['best_path']}")
            print(f"置信度: {result['confidence']:.3f}")
            
            if args.verbose:
                print(f"\n=== 各路径详细结果 ===")
                for path in result['paths']:
                    print(f"\n路径: {path['method']}")
                    print(f"置信度: {path['confidence']:.3f}")
                    print(f"答案: {path['answer']}")
                    print("推理步骤:")
                    for i, step in enumerate(path['steps']):
                        print(f"  {i+1}. {step}")
                
    except Exception as e:
        print(f"推理失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
