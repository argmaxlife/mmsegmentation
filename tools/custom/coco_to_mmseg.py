#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
import shutil
from collections import defaultdict

import cv2 as cv
import numpy as np
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# mmseg标签映射
CLASS2ID = {
    "background": 0,
    "sly_bjbmyw": 1,
    "sly_dmyw": 2,
}


def load_coco(coco_ann_path):
    """加载COCO标注"""
    with open(coco_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    img_id_to_info = {img["id"]: img for img in images}

    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    ann_by_img = defaultdict(list)
    for ann in annotations:
        ann_by_img[ann["image_id"]].append(ann)

    return img_id_to_info, ann_by_img, cat_id_to_name


def draw_polygon(mask, segmentation, class_id):
    """
    COCO polygon格式

    segmentation:
    [
        [x1,y1,x2,y2,...],
        [x1,y1,x2,y2,...]
    ]
    """
    for poly in segmentation:
        if len(poly) < 6:
            continue

        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)

        cv.fillPoly(mask, [pts], class_id)


def draw_rle(mask, segmentation, class_id):
    """
    COCO RLE格式
    """
    if mask_utils is None:
        raise ImportError("RLE格式需要安装pycocotools:\npip install pycocotools")

    decoded = mask_utils.decode(segmentation)

    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]

    mask[decoded > 0] = class_id


def get_image_size(img_info, img_dir):
    """
    优先从COCO中读取宽高
    """
    width = img_info.get("width")
    height = img_info.get("height")

    if width is not None and height is not None:
        return width, height

    img_path = os.path.join(img_dir, img_info["file_name"])

    img = cv.imread(img_path)
    if img is None:
        raise RuntimeError(f"读取图片失败: {img_path}")

    height, width = img.shape[:2]

    return width, height


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to MMSeg masks"
    )

    parser.add_argument("--coco_ann_path", required=True, help="COCO annotation json")

    parser.add_argument("--img_dir", required=True, help="image directory")

    parser.add_argument(
        "--mmseg_data_dir", required=True, help="output mmseg dataset directory"
    )

    parser.add_argument(
        "--ignore_empty",
        action="store_true",
        help="skip images without sly_bjbmyw or sly_dmyw",
    )

    args = parser.parse_args()

    output_img_dir = os.path.join(args.mmseg_data_dir, "images")

    output_ann_dir = os.path.join(args.mmseg_data_dir, "annotations")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)

    (
        img_id_to_info,
        ann_by_img,
        cat_id_to_name,
    ) = load_coco(args.coco_ann_path)

    instances = defaultdict(int)

    for img_id, img_info in tqdm(img_id_to_info.items(), desc="Converting"):
        width, height = get_image_size(img_info, args.img_dir)

        mask = np.zeros((height, width), dtype=np.uint8)

        anns = ann_by_img.get(img_id, [])

        has_target = False

        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"])

            if cat_name not in CLASS2ID:
                continue

            # background不算目标
            if cat_name in ("sly_bjbmyw", "sly_dmyw"):
                has_target = True

            class_id = CLASS2ID[cat_name]

            segmentation = ann.get("segmentation")
            if segmentation is None:
                continue

            if isinstance(segmentation, list):
                draw_polygon(mask, segmentation, class_id)

            elif isinstance(segmentation, dict):
                draw_rle(mask, segmentation, class_id)

            instances[cat_name] += 1

        # 与labelme脚本保持一致
        if args.ignore_empty and not has_target:
            continue

        file_name = Path(img_info["file_name"]).name

        src_img_path = os.path.join(args.img_dir, file_name)

        if not os.path.exists(src_img_path):
            print(f"WARNING: 图片不存在 {src_img_path}")
            continue

        basename = os.path.splitext(os.path.basename(file_name))[0]

        # 保存mask
        ann_path = os.path.join(output_ann_dir, f"{basename}.png")

        cv.imwrite(ann_path, mask)

        # 复制图片
        dst_img_path = os.path.join(output_img_dir, os.path.basename(file_name))

        shutil.copy2(src_img_path, dst_img_path)

    print("\nDone.")

    print("\nInstance statistics:")
    for cls_name, count in sorted(instances.items()):
        print(f"{cls_name}: {count}")


if __name__ == "__main__":
    main()
