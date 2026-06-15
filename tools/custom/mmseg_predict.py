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
        "--data-dir", "-d", type=str, required=True, help="data directory"
    )
    args = parser.parse_args()

    device = "cuda:1"
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
    colors = [(128, 128, 128), (0, 0, 255), (0, 255, 255)]

    input_dir = Path(args.data_dir)
    folder = input_dir.stem + "_pred"
    output_dir = checkpoint.parent / folder
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    tp = 0
    fp = 0
    fn = 0
    tp_bj = 0
    fp_bj = 0
    fp_bj_dm = 0
    fn_bj = 0
    tp_dm = 0
    fp_dm = 0
    fp_dm_bj = 0
    fn_dm = 0
    eps = 1e-8

    for file in tqdm(os.listdir(input_dir)):
        if not file.endswith(("jpg", "jpeg", "png")):
            continue

        impath = input_dir / file
        try:
            img = cv.imread(impath)
        except Exception as e:
            print(f"imread {file} exception: {e}")
            continue

        if img is None:
            print(f"imread return none: {file}")
            continue

        jsnpath = impath.with_suffix(".json")
        ann = None
        if jsnpath.exists():
            with open(jsnpath) as fps:
                ann = json.load(fps)

        seg_data = inference_model(model, img)
        sem_seg = seg_data.pred_sem_seg.cpu().data.numpy()[0]
        scores = torch.softmax(seg_data.seg_logits.data, dim=0).cpu().data.numpy()

        shapes = []
        pred_bjbmyw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pred_dmyw_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for cid, cname in enumerate(classes):
            if cname == "background":
                continue

            mask = sem_seg == cid
            mask = mask.astype(np.uint8)
            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[
                0
            ]
            ptss = []

            for contour in contours:
                pts = contour.astype(np.int32)
                ptss.append(pts)

                shape = {
                    "label": cname + "_pred",
                    "points": pts.reshape(-1, 2).tolist(),
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                }
                shapes.append(shape)

                contour_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                pts_ = contour.reshape(-1, 2).astype(np.int32)
                cv.fillPoly(contour_mask, [pts_], 255)
                slic = scores[cid, ...]
                pixels = slic[contour_mask > 0]
                score = np.mean(pixels) if pixels.size > 0 else 0
                org = np.mean(contour.reshape(-1, 2), axis=0).astype(np.int32)
                cv.putText(
                    img, f"{score:.2f}", org, cv.FONT_HERSHEY_COMPLEX, 1, colors[cid]
                )

            cv.polylines(img, ptss, True, colors[cid], 4)

            ptss = [pts.reshape(-1, 2) for pts in ptss]
            if cname == "sly_bjbmyw":
                cv.fillPoly(pred_bjbmyw_mask, ptss, 255)
            elif cname == "sly_dmyw":
                cv.fillPoly(pred_dmyw_mask, ptss, 255)

        has_fp_bj = False
        has_fp_dm = False
        has_fn_bj = False
        has_fn_dm = False
        if ann is not None:
            gt_bjbmyw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            gt_dmyw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            ptss_bj = []
            ptss_dm = []
            for shape in ann["shapes"]:
                cname = shape["label"]
                if cname not in ["sly_bjbmyw", "sly_dmyw"]:
                    continue

                cid = classes.index(cname)
                pts = np.array(shape["points"], dtype=np.int32).reshape(-1, 1, 2)
                cv.polylines(img, [pts], True, colors[cid], 1)

                pts = pts.reshape(-1, 2)
                if cname == "sly_bjbmyw":
                    ptss_bj.append(pts)
                elif cname == "sly_dmyw":
                    ptss_dm.append(pts)

            if len(ptss_bj) > 0:
                cv.fillPoly(gt_bjbmyw_mask, ptss_bj, 255)

            if len(ptss_dm) > 0:
                cv.fillPoly(gt_dmyw_mask, ptss_dm, 255)

            pred_mask = pred_bjbmyw_mask | pred_dmyw_mask
            gt_mask = gt_bjbmyw_mask | gt_dmyw_mask

            if np.count_nonzero(gt_mask) > 0:
                if np.count_nonzero(gt_mask & pred_mask) > 0:
                    tp += 1
                else:
                    fn += 1
            else:
                if np.count_nonzero(pred_mask) > 0:
                    fp += 1

            if np.count_nonzero(gt_bjbmyw_mask) > 0:
                if np.count_nonzero(gt_bjbmyw_mask & pred_bjbmyw_mask) > 0:
                    tp_bj += 1
                else:
                    fn_bj += 1
                    has_fn_bj = True
            else:
                if np.count_nonzero(pred_bjbmyw_mask) > 0:
                    fp_bj += 1
                    has_fp_bj = True

            if np.count_nonzero(gt_dmyw_mask) > 0:
                if np.count_nonzero(gt_dmyw_mask & pred_dmyw_mask) > 0:
                    tp_dm += 1
                else:
                    fn_dm += 1
                    has_fn_dm = True
            else:
                if np.count_nonzero(pred_dmyw_mask) > 0:
                    fp_dm += 1
                    has_fp_dm = True

            ann["shapes"].extend(shapes)
        else:
            pred_mask = pred_bjbmyw_mask | pred_dmyw_mask
            if np.count_nonzero(pred_mask) > 0:
                fp += 1

            if np.count_nonzero(pred_bjbmyw_mask) > 0:
                fp_bj += 1
                has_fp_bj = True

            if np.count_nonzero(pred_dmyw_mask) > 0:
                fp_dm += 1
                has_fp_dm = True

        if not has_fp_bj and not has_fp_dm and not has_fn_bj and not has_fn_dm:
            filname = output_dir / file
            jsnpath = output_dir / jsnpath.name
        else:
            fps = []
            if has_fp_bj:
                fps.append("bj")

            if has_fp_dm:
                fps.append("dm")

            fns = []
            if has_fn_bj:
                fns.append("bj")

            if has_fn_dm:
                fns.append("dm")

            mark_fp = ""
            if len(fps) > 0:
                mark_fp = "_fp_" + "_".join(fps)

            mark_fn = ""
            if len(fns) > 0:
                mark_fn = "_fn_" + "_".join(fns)

            mark = mark_fp + mark_fn
            basename, ext = os.path.splitext(file)
            filname = output_dir / f"{basename}{mark}{ext}"

            if ann is not None:
                basename, ext = os.path.splitext(jsnpath.name)
                jsnpath = output_dir / f"{basename}{mark}{ext}"
                ann["imagePath"] = filname.name

        cv.imwrite(filname, img)

        if ann is not None:
            with open(jsnpath, "w") as fps:
                json.dump(ann, fps, indent=2)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    precision_bj = tp_bj / (tp_bj + fp_bj + eps)
    recall_bj = tp_bj / (tp_bj + fn_bj + eps)

    precison_dm = tp_dm / (tp_dm + fp_dm + eps)
    recall_dm = tp_dm / (tp_dm + fn_dm + eps)

    metric_path = checkpoint.parent / f"{input_dir.stem}.txt"
    with open(metric_path, "a") as stream:
        stream.write(f"data = {args.data_dir}\n")
        stream.write(f"tp = {tp}, fp = {fp}, fn = {fn}\n")
        stream.write(f"precision = {precision}, recall = {recall}\n")
        stream.write(
            f"tp_bj = {tp_bj}, fp_bj = {fp_bj}, fp_bj_dm = {fp_bj_dm}, fn_bj = {fn_bj}\n"
        )
        stream.write(f"precision_bj = {precision_bj}, recall_bj = {recall_bj}\n")
        stream.write(
            f"tp_dm = {tp_dm}, fp_dm = {fp_dm}, fp_dm_bj = {fp_dm_bj}, fn_dm = {fn_dm}\n"
        )
        stream.write(f"precison_dm = {precison_dm}, recall_dm = {recall_dm}\n\n")


if __name__ == "__main__":
    main()
