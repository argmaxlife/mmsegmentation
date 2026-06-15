import argparse
from glob import glob
import os
from pathlib import Path
import random
import shutil
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="labelme to mmseg")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="labelme-format data directory",
    )
    args = parser.parse_args()

    k = 5
    data_dir = Path(args.data_dir)

    output_dir = Path(args.data_dir.rstrip("/") + "_kfold")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    jpgss = []
    for folder in os.listdir(data_dir):
        cd = data_dir / folder
        pattern = str(cd / "*.jpg")
        jpgs = glob(pattern)
        jpgs = [Path(jpg) for jpg in jpgs]
        jpgss.extend(jpgs)

    random.shuffle(jpgss)

    n = len(jpgss)
    size = n // k
    reminder = n % k
    sizes = [size] * (k - 1) + [size + reminder]
    print(f"kfold sizes: {sizes}")

    start = 0
    for i, size in tqdm(enumerate(sizes), leave=False):
        end = start + size
        jpgs = jpgss[start:end]
        print(f"start = {start}, end = {end}, jpgs = {len(jpgs)}")
        subdir = output_dir / f"{i}"
        subdir.mkdir(parents=True)
        for jpg in tqdm(jpgs):
            shutil.copy2(jpg, subdir)
            jsn = jpg.with_suffix(".json")
            shutil.copy2(jsn, subdir)
        start = end

    print("DONE")


if __name__ == "__main__":
    main()
