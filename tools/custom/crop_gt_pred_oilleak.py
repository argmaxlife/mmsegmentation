import os
import json
import cv2
import numpy as np
import argparse
from shapely.geometry import Polygon

def calculate_iou(poly1, poly2):
    # 确保poly2是一个有效的多边形
    if len(poly1) < 4 or len(poly2) < 4:
        print(f"Warning: Polygon has fewer than 4 points. Skipping IOU calculation for {poly2}")
        return 0

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)

    if poly1.is_valid and poly2.is_valid:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0
    return 0


def extract_rectangle_from_polygon(polygon):
    # 检查polygon是否有效且非空
    if len(polygon) < 3:
        print(f"Warning: Polygon has fewer than 3 points. Skipping boundingRect calculation for {polygon}")
        return 0, 0, 0, 0
    
    poly = np.array(polygon)
    
    # 确保多边形的点是float32类型，因为cv2.boundingRect要求该类型
    if poly.dtype != np.float32:
        poly = poly.astype(np.float32)
    
    # 计算边界矩形
    x, y, w, h = cv2.boundingRect(poly)
    return x, y, w, h


def process_json(input_dir, output_dir):
    # 创建output目录中的hit和miss子目录
    hit_dir = os.path.join(output_dir, 'hit')
    miss_dir = os.path.join(output_dir, 'miss')
    os.makedirs(hit_dir, exist_ok=True)
    os.makedirs(miss_dir, exist_ok=True)

    # 遍历input目录中的所有文件
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            print(file)
            json_file_path = os.path.join(input_dir, file)
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            
            img_name = file.replace('.json', '.jpg')
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            shapes = json_data['shapes']
            
            img_height, img_width = img.shape[:2]

            for i, shape in enumerate(shapes):
                if '_pred' not in shape['label'] and '_susp' not in shape['label']:
                    # 如果是人工标注
                    label = shape['label']
                    poly1 = shape['points']
                    max_iou = 0
                    best_match = None
                    best_match_idx = None
                    
                    for j, pred_shape in enumerate(shapes):
                        if '_pred' in pred_shape['label'] and '_susp' not in shape['label']:
                            poly2 = pred_shape['points']
                            iou = calculate_iou(poly1, poly2)
                            if iou > max_iou:
                                max_iou = iou
                                best_match = pred_shape
                                best_match_idx = j
                    
                    if best_match and max_iou > 0:
                        # 找到匹配的预测标签
                        # 计算并集的矩形区域
                        poly2 = best_match['points']
                        x1, y1, w1, h1 = extract_rectangle_from_polygon(poly1)
                        x2, y2, w2, h2 = extract_rectangle_from_polygon(poly2)
                        
                        x_min = min(x1, x2)
                        y_min = min(y1, y2)
                        x_max = max(x1 + w1, x2 + w2)
                        y_max = max(y1 + h1, y2 + h2)
                        
                        # 扩大矩形区域
                        dx = int((x_max - x_min) * 0.1)
                        dy = int((y_max - y_min) * 0.1)
                        
                        x_min -= dx
                        y_min -= dy
                        x_max += dx
                        y_max += dy
                        
                        # 确保坐标不会超出图像范围
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img_width, x_max)
                        y_max = min(img_height, y_max)
                        
                        # 裁剪图片
                        cropped_img = img[y_min:y_max, x_min:x_max]
                        
                        # 保存裁剪图
                        output_name = f"{img_name.replace('.jpg', '')}_{i}_{best_match_idx}_{label}.jpg"
                        output_path = os.path.join(hit_dir, output_name)
                        cv2.imwrite(output_path, cropped_img)
                    else:
                        # 如果没有找到匹配的预测标签
                        output_name = f"{img_name.replace('.jpg', '')}_{i}_miss_{label}.jpg"
                        output_path = os.path.join(miss_dir, output_name)
                        cv2.imwrite(output_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process LabelMe JSON files and generate cropped images.")
    parser.add_argument('--input-dir', type=str, required=True, help="Input directory containing JSON and image files.")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory to save the cropped images.")
    
    args = parser.parse_args()
    
    process_json(args.input_dir, args.output_dir)
