#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional
import requests
from PIL import Image
import io


def process_vision_info(messages: List[dict]) -> Tuple[List, List]:
    """处理视觉信息，提取图像和视频输入"""
    
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if "content" in message:
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            image_path = item.get("image")
                            if image_path:
                                # 检查是否为本地文件
                                if image_path.startswith(('http://', 'https://')):
                                    # 远程URL，尝试下载
                                    try:
                                        response = requests.get(image_path, timeout=10)
                                        response.raise_for_status()
                                        image = Image.open(io.BytesIO(response.content))
                                        image_inputs.append(image)
                                    except Exception as e:
                                        print(f"图像下载失败 {image_path}: {e}")
                                        # 如果下载失败，尝试直接使用URL
                                        image_inputs.append(image_path)
                                else:
                                    # 本地文件路径，直接加载
                                    try:
                                        image = Image.open(image_path).convert("RGB")
                                        image_inputs.append(image)
                                    except Exception as e:
                                        print(f"本地图像加载失败 {image_path}: {e}")
                                        # 如果加载失败，尝试直接使用路径
                                        image_inputs.append(image_path)
                        elif item.get("type") == "video":
                            video_url = item.get("video")
                            if video_url:
                                video_inputs.append(video_url)
    
    return image_inputs, video_inputs

