import argparse
from collections import defaultdict
import numpy as np
import re


def main():
    parser = argparse.ArgumentParser(description="find the best mmseg model")
    parser.add_argument("--log", "-l", type=str, help="log path")
    parser.add_argument("--metric", "-m", type=str, default="iou", help="metric")
    args = parser.parse_args()

    log_path = args.log
    with open(log_path) as fp:
        lines = fp.readlines()

    ious = defaultdict(list)
    accs = defaultdict(list)
    checkpoints = []

    for line in lines:
        content = line.strip()
        words = content.split()
        if "sly_bjbmyw" in words or "sly_dmyw" in words:
            ious[words[1]].append(float(words[3]))
            accs[words[1]].append(float(words[5]))
        elif "Saving checkpoint at" in content:
            iterations = re.search(r"at (\d+) iterations", content)
            iterations = iterations.group(1)
            checkpoint = f"iter_{iterations}.pth"
            checkpoints.append(checkpoint)

    assert len(ious["sly_bjbmyw"]) == len(ious["sly_dmyw"])
    assert len(accs["sly_bjbmyw"]) == len(accs["sly_dmyw"])

    metric = args.metric
    values = dict(iou=ious, acc=accs)

    metric_values = []
    for i in range(len(values[metric]["sly_bjbmyw"])):
        metric_value = (values[metric]["sly_bjbmyw"][i] + values[metric]["sly_dmyw"][i]) * 0.5
        metric_values.append(metric_value)

    metric_values = np.array(metric_values, dtype=np.float32)

    best_idx = np.argmax(metric_values)
    best_metric_value = metric_values[best_idx]
    best_ckpt = checkpoints[best_idx]

    print(f"best_idx = {best_idx}, best ckpt = {best_ckpt}, best {metric} = {best_metric_value}")


if __name__ == "__main__":
    main()
