import argparse
import cv2 as cv
import json
import numpy as np
import os
from pathlib import Path
import shutil
import sys
from tqdm import tqdm

import torch

sys.path.insert(0, os.getcwd())
from mmseg.apis import inference_model, init_model


def main():
    parser = argparse.ArgumentParser(description="mmseg prediction")
    parser.add_argument(
        "--checkpoint", "-ckt", type=str, required=True, help="checkppoint path"
    )
    parser.add_argument("--config", "-cfg", type=str, required=True, help="config file")
    parser.add_argument(
        "--img-dir", "-d", type=str, required=True, help="image directory"
    )
    args = parser.parse_args()

    device = "cuda:0"
    config = args.config
    checkpoint = Path(args.checkpoint)
    pure_checkpoint = checkpoint.parent / (
        checkpoint.stem + "_pure" + checkpoint.suffix
    )
    if not pure_checkpoint.exists():
        try:
            model_data = torch.load(checkpoint, weights_only=False)
        except Exception as e:
            print(f"torch.load exception: {e}, try again")
            model_data = torch.load(checkpoint)
        obj = {"meta": model_data["meta"], "state_dict": model_data["state_dict"]}
        torch.save(obj, pure_checkpoint)

    model = init_model(config, str(pure_checkpoint), device=device)
    classes = ["background", "sly_bjbmyw", "sly_dmyw"]
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255)]

    img_dir = Path(args.img_dir)
    folder = img_dir.stem + "_labeled"
    output_dir = checkpoint.parent / folder
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    for jpg in tqdm(img_dir.glob("*.jpg")):
        try:
            img = cv.imread(jpg)
        except Exception as e:
            print(f"imread {jpg} exception: {e}")
            continue

        seg_data = inference_model(model, img)
        sem_seg = seg_data.pred_sem_seg.cpu().data.numpy()[0]
        scores = torch.softmax(seg_data.seg_logits.data, dim=0).cpu().data.numpy()
        shapes = []

        for cid, cname in enumerate(classes):
            if cname == "background":
                continue

            mask = sem_seg == cid
            mask = mask.astype(np.uint8)
            contours, hierarchy = cv.findContours(
                mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )
            ptss = []
            background_ptss = []

            for contour_idx, contour in enumerate(contours):
                pts = contour.astype(np.int32)
                label = cname
                inner_ptss = []
                if hierarchy[0][contour_idx][3] != -1:
                    label = "background"
                    background_ptss.append(pts)
                else:
                    ptss.append(pts)
                    for q_contour_idx, q_contour in enumerate(contours):
                        if hierarchy[0][q_contour_idx][3] == contour_idx:
                            inner_ptss.append(q_contour.astype(np.int32).reshape(-1, 2))

                approx = cv.approxPolyDP(contour, 1, True)
                approx = approx.astype(np.int32).reshape(-1, 2)
                shape = {
                    "label": label,
                    "points": approx.tolist(),
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                }
                shapes.append(shape)

                if label == "background":
                    continue

                points = pts.reshape(-1, 2)
                contour_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv.fillPoly(contour_mask, [points], 255)
                if len(inner_ptss) > 0:
                    cv.fillPoly(contour_mask, inner_ptss, 0)

                score_slice = scores[cid, ...]
                pixels = score_slice[contour_mask > 0]
                score = np.mean(pixels) if pixels.size > 0 else 0
                org = np.mean(contour.reshape(-1, 2), axis=0).astype(np.int32)
                cv.putText(
                    img, f"{score:.2f}", org, cv.FONT_HERSHEY_COMPLEX, 1, colors[cid]
                )

            cv.polylines(img, ptss, True, colors[cid], 4)
            if len(background_ptss) > 0:
                cv.polylines(img, background_ptss, True, colors[0], 4)

        output_jpg = output_dir / jpg.name
        cv.imwrite(output_jpg, img)

        ann = {
            "version": "5.8.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": jpg.name,
            "imageData": None,
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1],
        }

        output_json = output_jpg.with_suffix(".json")
        with open(output_json, "w") as fps:
            json.dump(ann, fps, indent=2)


if __name__ == "__main__":
    main()
