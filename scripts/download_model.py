#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from modelscope import snapshot_download

def download_qwen_model():
    """从ModelScope下载Qwen2.5-VL-3B-Instruct模型"""
    
    print("开始从ModelScope下载Qwen2.5-VL-3B-Instruct模型...")
    
    try:
        # 设置下载目录
        local_dir = "./models/Qwen2.5-VL-3B-Instruct"
        
        # 从ModelScope下载模型
        model_dir = snapshot_download(
            model_id="qwen/Qwen2.5-VL-3B-Instruct",
            cache_dir=local_dir,
            revision="master"
        )
        
        print(f"模型下载完成！")
        print(f"模型路径: {model_dir}")
        
        # 检查文件大小
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        
        print(f"文件数量: {file_count}")
        print(f"总大小: {total_size / (1024**3):.2f} GB")
        
        return model_dir
        
    except Exception as e:
        print(f"下载失败: {e}")
        return None

if __name__ == "__main__":
    model_path = download_qwen_model()
    if model_path:
        print(f"\n下载成功！模型保存在: {model_path}")
        print("现在可以使用以下命令运行推理:")
        print(f"python src/infer_cot.py --model-id {model_path} --image your_image.jpg --question 'your question'")
    else:
        print("下载失败，请检查网络连接或重试")
