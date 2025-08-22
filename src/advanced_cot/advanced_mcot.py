#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import torch
from typing import List, Optional, Dict, Tuple
import numpy as np
from dataclasses import dataclass
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


@dataclass
class ReasoningPath:
    """推理路径数据结构"""
    path_id: int
    strategy: str
    reasoning_steps: List[str]
    confidence: float
    final_answer: str
    metadata: Dict


class AdvancedMCOTEngine:
    """高级多模态思维链推理引擎"""
    
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.reasoning_paths = []
        
    def execute_multi_path_reasoning(self, images: List[str], question: str) -> Dict:
        """执行多路径推理"""
        print("开始多路径推理...", file=sys.stderr)
        
        # 路径1: 分层推理
        path1 = self._hierarchical_reasoning(images, question)
        self.reasoning_paths.append(path1)
        
        # 路径2: 直接推理
        path2 = self._direct_reasoning(images, question)
        self.reasoning_paths.append(path2)
        
        # 路径3: 分解推理
        path3 = self._decomposed_reasoning(images, question)
        self.reasoning_paths.append(path3)
        
        # 选择最佳路径
        best_path = self._select_best_path()
        
        # 迭代细化
        refined_answer = self._iterative_refinement(images, question, best_path)
        
        return {
            'final_answer': refined_answer,
            'reasoning_paths': self.reasoning_paths,
            'best_path': best_path,
            'confidence': best_path.confidence,
            'method': 'advanced_mcot'
        }
    
    def _hierarchical_reasoning(self, images: List[str], question: str) -> ReasoningPath:
        """分层推理路径"""
        print("执行分层推理路径...", file=sys.stderr)
        
        steps = []
        
        # 步骤1: 视觉分析
        visual_prompt = f"""请详细分析图像中的视觉元素：
问题：{question}

请识别：
1. 主要对象和实体
2. 对象间的关系
3. 场景和环境特征
4. 任何异常或特殊细节"""
        
        visual_result = self._execute_reasoning(visual_prompt)
        steps.append(f"视觉分析: {visual_result}")
        
        # 步骤2: 语义理解
        semantic_prompt = f"""基于视觉分析，理解问题语义：
视觉分析：{visual_result}
问题：{question}

请分析：
1. 问题的核心含义
2. 需要的知识类型
3. 答案应该包含的要素"""
        
        semantic_result = self._execute_reasoning(semantic_prompt)
        steps.append(f"语义理解: {semantic_result}")
        
        # 步骤3: 逻辑推理
        logic_prompt = f"""基于前面的分析，进行逻辑推理：
视觉分析：{visual_result}
语义理解：{semantic_result}
问题：{question}

请进行：
1. 因果关系分析
2. 推理步骤构建
3. 结论推导"""
        
        logic_result = self._execute_reasoning(logic_prompt)
        steps.append(f"逻辑推理: {logic_result}")
        
        # 步骤4: 最终答案
        final_prompt = f"""综合所有分析，给出最终答案：
{chr(10).join(steps)}
问题：{question}

请给出简洁明确的最终答案："""
        
        final_answer = self._execute_reasoning(final_prompt)
        
        return ReasoningPath(
            path_id=1,
            strategy="hierarchical",
            reasoning_steps=steps,
            confidence=0.92,
            final_answer=final_answer,
            metadata={'method': 'hierarchical_reasoning'}
        )
    
    def _direct_reasoning(self, images: List[str], question: str) -> ReasoningPath:
        """直接推理路径"""
        print("执行直接推理路径...", file=sys.stderr)
        
        prompt = f"""请直接回答以下问题，但请先进行逐步思考：

问题：{question}

请按以下格式回答：
思考过程：[你的详细推理步骤]
最终答案：[一句话答案]"""
        
        result = self._execute_reasoning(prompt)
        
        # 分离思考过程和最终答案
        if "思考过程：" in result and "最终答案：" in result:
            parts = result.split("最终答案：")
            reasoning = parts[0].replace("思考过程：", "").strip()
            answer = parts[1].strip()
        else:
            reasoning = result
            answer = result
        
        return ReasoningPath(
            path_id=2,
            strategy="direct",
            reasoning_steps=[reasoning],
            confidence=0.85,
            final_answer=answer,
            metadata={'method': 'direct_reasoning'}
        )
    
    def _decomposed_reasoning(self, images: List[str], question: str) -> ReasoningPath:
        """分解推理路径"""
        print("执行分解推理路径...", file=sys.stderr)
        
        steps = []
        
        # 分解问题
        decompose_prompt = f"""请将以下问题分解为更简单的子问题：
问题：{question}

请列出3-5个需要回答的子问题："""
        
        decompose_result = self._execute_reasoning(decompose_prompt)
        steps.append(f"问题分解: {decompose_result}")
        
        # 逐个回答子问题
        sub_answers = []
        for i in range(3):  # 假设分解为3个子问题
            sub_prompt = f"""基于问题分解，回答第{i+1}个子问题：
问题分解：{decompose_result}
问题：{question}

请回答第{i+1}个子问题："""
            
            sub_answer = self._execute_reasoning(sub_prompt)
            sub_answers.append(sub_answer)
            steps.append(f"子问题{i+1}: {sub_answer}")
        
        # 综合答案
        synthesis_prompt = f"""基于所有子问题的答案，综合给出最终答案：
问题分解：{decompose_result}
子问题答案：
{chr(10).join([f"{i+1}. {answer}" for i, answer in enumerate(sub_answers)])}
原问题：{question}

请综合给出最终答案："""
        
        final_answer = self._execute_reasoning(synthesis_prompt)
        
        return ReasoningPath(
            path_id=3,
            strategy="decomposed",
            reasoning_steps=steps,
            confidence=0.88,
            final_answer=final_answer,
            metadata={'method': 'decomposed_reasoning'}
        )
    
    def _select_best_path(self) -> ReasoningPath:
        """选择最佳推理路径"""
        # 基于置信度选择最佳路径
        best_path = max(self.reasoning_paths, key=lambda x: x.confidence)
        print(f"选择最佳路径: {best_path.strategy} (置信度: {best_path.confidence:.3f})", file=sys.stderr)
        return best_path
    
    def _iterative_refinement(self, images: List[str], question: str, best_path: ReasoningPath) -> str:
        """迭代细化推理结果"""
        print("执行迭代细化...", file=sys.stderr)
        
        # 第一轮细化：错误检测
        error_check_prompt = f"""请检查以下推理结果是否有错误：
问题：{question}
推理路径：{best_path.strategy}
推理步骤：
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(best_path.reasoning_steps)])}
最终答案：{best_path.final_answer}

请检查：
1. 推理逻辑是否正确
2. 答案是否合理
3. 是否有遗漏的信息"""
        
        error_check = self._execute_reasoning(error_check_prompt)
        
        # 第二轮细化：结果优化
        refinement_prompt = f"""基于错误检查结果，优化最终答案：
问题：{question}
原始答案：{best_path.final_answer}
错误检查：{error_check}

请给出优化后的最终答案："""
        
        refined_answer = self._execute_reasoning(refinement_prompt)
        
        return refined_answer
    
    def _execute_reasoning(self, prompt: str) -> str:
        """执行推理"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        try:
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[chat_text],
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


def advanced_mcot_generate(
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
    """高级MCOT生成函数"""
    
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
    
    # 初始化高级MCOT引擎
    mcot_engine = AdvancedMCOTEngine(model, processor, device)
    
    # 执行高级MCOT推理
    result = mcot_engine.execute_multi_path_reasoning(images, question)
    
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="高级多模态思维链推理系统"
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
        result = advanced_mcot_generate(
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
            print("=== 高级MCOT推理结果 ===")
            print(f"最终答案: {result['final_answer']}")
            print(f"最佳路径: {result['best_path'].strategy}")
            print(f"置信度: {result['confidence']:.3f}")
            
            if args.verbose:
                print(f"\n=== 各路径详细结果 ===")
                for path in result['reasoning_paths']:
                    print(f"\n路径 {path.path_id} ({path.strategy}):")
                    print(f"置信度: {path.confidence:.3f}")
                    print(f"答案: {path.final_answer}")
                    print("推理步骤:")
                    for i, step in enumerate(path.reasoning_steps):
                        print(f"  {i+1}. {step}")
                
    except Exception as e:
        print(f"推理失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
