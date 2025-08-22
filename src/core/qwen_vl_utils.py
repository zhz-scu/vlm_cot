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
                            image_url = item.get("image")
                            if image_url:
                                try:
                                    # 下载图像
                                    response = requests.get(image_url, timeout=10)
                                    response.raise_for_status()
                                    image = Image.open(io.BytesIO(response.content))
                                    image_inputs.append(image)
                                except Exception as e:
                                    print(f"图像下载失败 {image_url}: {e}")
                                    # 如果下载失败，尝试直接使用URL
                                    image_inputs.append(image_url)
                        elif item.get("type") == "video":
                            video_url = item.get("video")
                            if video_url:
                                video_inputs.append(video_url)
    
    return image_inputs, video_inputs

