import argparse
import cv2 as cv
import json
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="sort false positive")
    parser.add_argument(
        "--prepared-data-dir", "-pd", type=str, help="prepared data dir"
    )
    parser.add_argument(
        "-sort-fp-dir", "-sf", type=str, help="sorted false positive dir"
    )
    parser.add_argument(
        "--labelme-data-dir", "-ldd", type=str, help="labelme format data dir"
    )
    args = parser.parse_args()

    prepared_data_dir = Path(args.prepared_data_dir)
    sort_fp_dir = Path(args.sort_fp_dir)

    net_input_dir = prepared_data_dir / "net_input"
    gt_masks_dir = prepared_data_dir / "gt_masks"
    inference_output_dir = prepared_data_dir / "inference_output"
    logits_dir = prepared_data_dir / "logits"

    if sort_fp_dir.exists():
        shutil.rmtree(sort_fp_dir)
    sort_fp_dir.mkdir(parents=True)

    labelme_data_dir = Path(args.labelme_data_dir)
    jpgs = labelme_data_dir.glob("*/*.jpg")
    stem2jpg = {jpg.stem: jpg for jpg in jpgs}

    sort_labelme_data_dir = labelme_data_dir.parent / "sorted_labelme"
    if sort_labelme_data_dir.exists():
        shutil.rmtree(sort_labelme_data_dir)
    sort_labelme_data_dir.mkdir(parents=True)

    seg_info = []
    cid = 1
    classes = ["background", "sly_bjbmyw", "sly_dmyw"]

    for input_png in tqdm(list(net_input_dir.glob("*.png"))):
        stem = input_png.stem
        gt_png = gt_masks_dir / f"{stem}.png"
        if not gt_png.exists():
            print(f"{gt_png} does not exists")
            continue

        output_png = inference_output_dir / f"{stem}.png"
        if not output_png.exists():
            print(f"{output_png} does not exists")
            continue

        logits_npy = logits_dir / f"{stem}.npy"
        if not logits_npy.exists():
            print(f"{logits_npy} does not exists")
            continue

        gt_img = cv.imread(gt_png, cv.IMREAD_GRAYSCALE)
        output_img = cv.imread(output_png, cv.IMREAD_GRAYSCALE)
        with open(logits_npy, "rb") as fp:
            logits = np.load(fp)

        pred_mask = output_img == cid
        pred_mask = pred_mask.astype(np.uint8)
        gt_mask = gt_img == cid
        gt_mask = gt_mask.astype(np.uint8) * 255

        logits = logits.transpose(0, 2, 1)
        scores = softmax(logits)[1, ...]

        contours, _ = cv.findContours(
            pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        local_mask = np.zeros_like(pred_mask)
        fps = []

        for contour_idx, contour in enumerate(contours):
            points = contour.astype(np.int32).reshape(-1, 2)
            cv.fillPoly(local_mask, [points], 255)

            hits = np.count_nonzero(gt_mask & local_mask)
            if hits > 0:
                local_mask[...] = 0
                continue

            pixels = scores[local_mask > 0]
            score = np.mean(pixels)
            fps.append([contour_idx, score])
            local_mask[...] = 0

        sorted_fps = list(sorted(fps, key=lambda fp: fp[1]))

        global_score = sorted_fps[-1][1] if len(sorted_fps) > 0 else 0
        fp_contours = [contours[fp[0]] for fp in fps]

        seg_info.append([input_png, global_score, fp_contours])

    sorted_seg_info = list(sorted(seg_info, key=lambda si: -si[1]))

    for idx, si in enumerate(tqdm(sorted_seg_info)):
        input_img = cv.imread(si[0])
        contours = si[2]

        pts = []
        shapes = []
        for contour in contours:
            pts.append(contour.astype(np.int32))

            approx = cv.approxPolyDP(contour, 1, True)
            approx = approx.astype(np.int32).reshape(-1, 2)
            shape = {
                "label": f"{classes[cid]}_pred",
                "points": approx.tolist(),
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
            shapes.append(shape)

        if len(pts) > 0:
            cv.polylines(input_img, pts, True, (0, 255, 255), 3)

        sorted_png = sort_fp_dir / f"{idx:08d}_{si[0].name}"
        cv.imwrite(sorted_png, input_img)

        jpg = stem2jpg[si[0].stem]
        sorted_jpg = sort_labelme_data_dir / f"{idx:08d}_{jpg.name}"
        shutil.copyfile(jpg, sorted_jpg)

        jsn = jpg.with_suffix(".json")
        with open(jsn) as stream:
            ann = json.load(stream)

        ann["shapes"].extend(shapes)
        ann["imagePath"] = sorted_jpg.name
        sorted_jsn = sorted_jpg.with_suffix(".json")
        with open(sorted_jsn, "w") as stream:
            json.dump(ann, stream, indent=2)


if __name__ == "__main__":
    main()
