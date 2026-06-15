#!/usr/bin/env python3
# detect_and_copy_extended.py

import argparse
import shutil
import json
from pathlib import Path
from ultralytics import YOLO  # pip install ultralytics
from shapely.geometry import Polygon, box
from tqdm import tqdm  # pip install tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 YOLOv8 检测图片中的指定类别并检查与特定多边形的相交，然后复制文件"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="YOLOv8 模型权重文件路径，例如 yolov8n.pt",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="输入目录，包含 train/val 子目录",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="输出目录，将在此目录下创建 train/val 结构",
    )
    parser.add_argument(
        "--class-names",
        "-c",
        nargs="+",
        required=True,
        help="要检测的类别名称列表，例如 -c thing cat dog",
    )
    parser.add_argument(
        "--poly-label",
        "-p",
        type=str,
        default="sly",
        help="要解析并检测相交的多边形标签，默认为 'sly'",
    )
    return parser.parse_args()


def load_sly_polygons(json_path, poly_label):
    """
    从 LabelMe 格式的 JSON 读取所有标签为 poly_label 的多边形
    返回 shapely.geometry.Polygon 对象列表
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polys = []
    for shape in data.get("shapes", []):
        if shape.get("label") == poly_label and shape.get("shape_type") == "polygon":
            pts = shape.get("points", [])
            if pts:
                poly = Polygon(pts)
                if poly.is_valid:
                    polys.append(poly)
    return polys


def main():
    args = parse_args()

    # 加载 YOLO 模型
    model = YOLO(str(args.model))

    # 创建输出目录结构
    args.output.mkdir(parents=True, exist_ok=True)

    subsets = ["train", "val"]
    # 外层进度条：子目录处理
    for subset in tqdm(subsets, desc="Processing subset"):
        in_subdir = args.input / subset
        out_subdir = args.output / subset
        out_subdir.mkdir(parents=True, exist_ok=True)

        img_files = list(in_subdir.rglob("*.jpg"))
        # 内层进度条：图片文件处理
        for img_path in tqdm(img_files, desc=f"  Images ({subset})", leave=False):
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue

            # 检测图片
            results = model.predict(source=str(img_path), save=False, verbose=False)
            det = results[0]

            # 构建检测到的指定类别的框列表
            thing_boxes = []
            names = det.names
            for box_data, cls_idx in zip(
                det.boxes.xyxy.tolist(), det.boxes.cls.tolist()
            ):
                label = names[int(cls_idx)]
                if label in args.class_names:
                    x1, y1, x2, y2 = box_data
                    thing_boxes.append(box(x1, y1, x2, y2))

            if not thing_boxes:
                continue

            # 解析多边形
            sly_polys = load_sly_polygons(json_path, args.poly_label)
            if not sly_polys:
                continue

            # 判断任一多边形与任一检测框是否相交
            intersect_found = any(
                poly.intersects(tb) for poly in sly_polys for tb in thing_boxes
            )

            # 相交则复制文件
            if intersect_found:
                shutil.copy2(img_path, out_subdir / img_path.name)
                shutil.copy2(json_path, out_subdir / json_path.name)
                print(
                    f"Copied {subset}/{img_path.name} and its JSON due to intersection."
                )


if __name__ == "__main__":
    main()
