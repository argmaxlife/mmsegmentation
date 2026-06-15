import os
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def detect_and_copy(model, input_dir, output_dir, classes):
    # 创建output目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取input目录下所有jpg文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

    # 使用tqdm添加进度条
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)

        # 检测图片
        results = model(image_path, verbose=False, conf=0.5)

        # 获取检测到的类别索引
        detected_class_indices = results[0].boxes.cls.cpu().numpy().astype(int)

        # 获取检测到的类别名称
        detected_classes = [results[0].names[idx] for idx in detected_class_indices]

        # 如果检测到目标类别，复制图片到output目录
        if any(cls in detected_classes for cls in classes):
            output_path = os.path.join(output_dir, image_file)
            shutil.copy(image_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Use YOLOv8 to detect and copy images")
    parser.add_argument("--model", required=True, help="YOLOv8模型路径")
    parser.add_argument("--input", required=True, help="输入图片目录")
    parser.add_argument("--output", required=True, help="输出图片目录")
    parser.add_argument(
        "--classes", required=True, nargs="+", help="检测目标类别名称列表"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载YOLOv8模型
    model = YOLO(args.model)

    # 调用检测和复制功能
    detect_and_copy(model, args.input, args.output, args.classes)
