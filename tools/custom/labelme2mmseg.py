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
        "--data_dir", "-d", type=str, required=True, help="data directory"
    )
    parser.add_argument("--classes", "-c", type=str, nargs="+", help="classes")
    args = parser.parse_args()

    data_dir = args.data_dir
    srcs = ["train", "val"]
    dsts = ["training", "validation"]
    classes = args.classes or ["sly_bjbmyw", "sly_dmyw"]
    print(f"classes: {classes}")
    stat_path = os.path.join(data_dir, os.path.pardir, "mmseg.stat")
    ostream = open(stat_path, "w")
    ostream.write("subset\tclass\tinstances\n")

    for src, dst in zip(srcs, dsts):
        input_dir = os.path.join(data_dir, src)
        output_img_dir = os.path.join(data_dir, os.path.pardir, "mmseg", "images", dst)
        output_ann_dir = os.path.join(
            data_dir, os.path.pardir, "mmseg", "annotations", dst
        )
        os.makedirs(output_img_dir)
        os.makedirs(output_ann_dir)
        instances = defaultdict(int)

        for imfile in tqdm(os.listdir(input_dir)):
            if imfile.endswith(".json"):
                continue

            basename = os.path.splitext(imfile)[0]
            jspath = os.path.join(input_dir, f"{basename}.json")
            if not os.path.exists(jspath):
                continue

            with open(jspath) as fp:
                ann = json.load(fp)

            shapes = ann["shapes"]
            yws = [shape for shape in shapes if shape["label"] in classes]

            if len(yws) == 0:
                continue

            img_path = os.path.join(input_dir, imfile)
            shutil.copy2(img_path, output_img_dir)

            width, height = imagesize.get(img_path)
            label = np.zeros((height, width), dtype=np.uint8)

            backgrounds = []
            for shape in ann["shapes"]:
                if shape["label"] not in classes:
                    if shape["label"] == "background":
                        backgrounds.append(shape)
                    continue

                points = np.array(shape["points"]).round().astype(np.int32)
                points = points.reshape(-1, 2)
                if points.shape[0] < 3:
                    continue

                color = 0
                if shape["label"] == "sly_bjbmyw":
                    color = 1
                elif shape["label"] == "sly_dmyw":
                    color = 2
                elif shape["label"] == "sly_bjbmyw_slight":
                    color = 1

                cv.fillPoly(label, [points], color)
                instances[shape["label"]] += 1

            color = 0
            for shape in backgrounds:
                pts = np.array(shape["points"], dtype=np.int32)
                cv.fillPoly(label, [pts], color)

            filename = os.path.join(output_ann_dir, f"{basename}.png")
            cv.imwrite(filename, label)

        for k, v in instances.items():
            ostream.write(f"{dst}\t{k}\t{v}\n")

    ostream.close()


if __name__ == "__main__":
    main()
