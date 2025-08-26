#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
from typing import List, Dict
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from improved_cos_model import ObjectDetector, DetectedRegion


def ensure_image(img_path: str) -> Image.Image:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像不存在: {img_path}")
    return Image.open(img_path).convert("RGB")


def visualize(image: Image.Image, regions: List[DetectedRegion], out_path: str):
    from improved_cos_model import _create_visualization
    vis = _create_visualization(image, regions)
    vis.save(out_path)


def run_detector(detector_type: str, image: Image.Image, min_conf: float) -> Dict:
    det = ObjectDetector("cpu")
    det.initialize_detector(detector_type)
    if not det.initialized:
        return {"ok": False, "error": f"{detector_type} 初始化失败"}
    t0 = time.time()
    regions = det.detect_regions(image, min_confidence=min_conf)
    dt = time.time() - t0
    return {
        "ok": True,
        "type": detector_type,
        "time_s": dt,
        "regions": [
            {
                "bbox": r.bbox,
                "conf": r.confidence,
                "label": r.label,
                "region_id": r.region_id,
            } for r in regions
        ]
    }


def benchmark(image_path: str, min_conf: float = 0.3, out_dir: str = ".") -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    image = ensure_image(image_path)

    results = {}

    # 1) 传统CV：边缘检测
    edge_res = run_detector("edge", image, min_conf)
    results["edge"] = edge_res
    if edge_res.get("ok"):
        visualize(image, [DetectedRegion(bbox=r["bbox"], confidence=r["conf"], label=r.get("label", ""), content="", region_id=r["region_id"]) for r in edge_res["regions"]], os.path.join(out_dir, "benchmark_edge.png"))

    # 2) 深度学习：YOLO（若未安装会自动降级）
    yolo_res = run_detector("yolo", image, min_conf)
    results["yolo"] = yolo_res
    if yolo_res.get("ok"):
        visualize(image, [DetectedRegion(bbox=r["bbox"], confidence=r["conf"], label=r.get("label", ""), content="", region_id=r["region_id"]) for r in yolo_res["regions"]], os.path.join(out_dir, "benchmark_yolo.png"))

    # 汇总
    summary = {
        "image": image_path,
        "min_conf": min_conf,
        "edge": {
            "ok": edge_res.get("ok"),
            "num_regions": len(edge_res.get("regions", [])) if edge_res.get("ok") else 0,
            "time_s": edge_res.get("time_s"),
            "error": edge_res.get("error"),
        },
        "yolo": {
            "ok": yolo_res.get("ok"),
            "num_regions": len(yolo_res.get("regions", [])) if yolo_res.get("ok") else 0,
            "time_s": yolo_res.get("time_s"),
            "error": yolo_res.get("error"),
        },
    }

    with open(os.path.join(out_dir, "benchmark_detectors_result.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": results}, f, ensure_ascii=False, indent=2)

    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(description="传统CV vs 深度学习CV检测对比")
    ap.add_argument("--image", type=str, default="test_simple.png")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    summary = benchmark(args.image, args.min_conf, args.out_dir)

    print("=== 检测对比结果 ===")
    print(f"图像: {summary['image']}")
    print(f"阈值: {summary['min_conf']}")
    print("- 边缘检测:")
    print(f"  - 状态: {'OK' if summary['edge']['ok'] else 'FAIL'}")
    print(f"  - 区域数: {summary['edge']['num_regions']}")
    print(f"  - 耗时: {summary['edge']['time_s']}")
    if summary['edge'].get('error'): print(f"  - 错误: {summary['edge']['error']}")
    print("- YOLO:")
    print(f"  - 状态: {'OK' if summary['yolo']['ok'] else 'FAIL'}")
    print(f"  - 区域数: {summary['yolo']['num_regions']}")
    print(f"  - 耗时: {summary['yolo']['time_s']}")
    if summary['yolo'].get('error'): print(f"  - 错误: {summary['yolo']['error']}")
    print("可视化: benchmark_edge.png, benchmark_yolo.png")
    print("详情: benchmark_detectors_result.json")


if __name__ == "__main__":
    main()
