import argparse
from collections import defaultdict
import cv2 as cv
import imagesize
import json
import numpy as np
import os
import shutil
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="labelme to mmseg")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="labelme-format kfold data directory",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    folders = os.listdir(data_dir)
    n = len(folders)

    for i in range(n):
        srcs = []
        dsts = []
        for j in range(n):
            srcs.append(str(j))
            if j == i:
                dsts.append("validation")
            else:
                dsts.append("training")

        print(f"k{i}, srcs = {srcs}, dsts = {dsts}")
        counter = defaultdict(int)

        for src, dst in zip(srcs, dsts):
            input_dir = os.path.join(data_dir, src)
            output_img_dir = os.path.join(
                data_dir, os.path.pardir, "mmseg_kfold", str(i), "images", dst
            )
            output_ann_dir = os.path.join(
                data_dir, os.path.pardir, "mmseg_kfold", str(i), "annotations", dst
            )

            if not os.path.exists(output_img_dir):
                os.makedirs(output_img_dir)
            if not os.path.exists(output_ann_dir):
                os.makedirs(output_ann_dir)

            print(f"input_dir: {input_dir}")
            print(f"output_img_dir: {output_img_dir}")
            print(f"output_ann_dir: {output_ann_dir}")
            for imfile in tqdm(os.listdir(input_dir)):
                if imfile.endswith(".json"):
                    continue

                basename = os.path.splitext(imfile)[0]
                with open(os.path.join(input_dir, f"{basename}.json")) as fp:
                    ann = json.load(fp)

                shapes = ann["shapes"]
                yws = [
                    shape
                    for shape in shapes
                    if shape["label"] in ["sly_bjbmyw", "sly_dmyw"]
                ]

                if len(yws) == 0:
                    continue

                img_path = os.path.join(input_dir, imfile)
                shutil.copy2(img_path, output_img_dir)
                counter[dst] += 1

                width, height = imagesize.get(img_path)
                label = np.zeros((height, width), dtype=np.uint8)

                backgrounds = []
                for shape in ann["shapes"]:
                    if shape["label"] not in ["sly_bjbmyw", "sly_dmyw"]:
                        if shape["label"] == "background":
                            backgrounds.append(shape)
                        continue

                    color = 0
                    if shape["label"] == "sly_bjbmyw":
                        color = 1
                    elif shape["label"] == "sly_dmyw":
                        color = 2

                    pts = np.array(shape["points"], dtype=np.int32)
                    cv.fillPoly(label, [pts], color)

                color = 0
                for shape in backgrounds:
                    pts = np.array(shape["points"], dtype=np.int32)
                    cv.fillPoly(label, [pts], color)

                filename = os.path.join(output_ann_dir, f"{basename}.png")
                cv.imwrite(filename, label)

        print(counter)


if __name__ == "__main__":
    main()
