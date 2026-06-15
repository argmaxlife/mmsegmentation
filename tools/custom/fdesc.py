import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple


def encode(contour: np.ndarray, keep=16) -> Tuple[np.ndarray, np.ndarray, float, int]:
    center = np.mean(contour, axis=0)
    centered = contour - center
    norm = np.linalg.norm(centered).item()
    s = 0
    for e in centered:
        s += (e[0]*e[0]+e[1]*e[1])
    s = s**0.5
    centered = np.expand_dims(centered, 1)
    dft = cv2.dft(centered, flags=cv2.DFT_COMPLEX_OUTPUT)
    hkeep = keep // 2
    fdesc = np.zeros((keep * 2,), dtype=contour.dtype)
    fdesc[:keep] = dft[:hkeep].reshape(-1)
    fdesc[-keep:] = dft[-hkeep:].reshape(-1)
    return (fdesc, center, norm, contour.shape[0])


def decode(encoded: Tuple[np.ndarray, np.ndarray, float, int]) -> np.ndarray:
    fdesc, center, norm, n = encoded
    keep = fdesc.shape[0] // 2
    hkeep = keep // 2
    dft_truncated = np.zeros((n, 1, 2), dtype=fdesc.dtype)
    dft_truncated[:hkeep] = fdesc[:keep].reshape(-1, 1, 2)
    dft_truncated[-hkeep:] = fdesc[-keep:].reshape(-1, 1, 2)
    idft = cv2.idft(dft_truncated, flags=cv2.DFT_COMPLEX_OUTPUT)
    decoded = np.squeeze(idft, 1)
    scale = norm / np.linalg.norm(decoded)
    decoded *= scale
    decoded += center
    return decoded


if __name__ == "__main__":
    img_dir = Path("../../data/substation/expander/v9.0/train_crop_crop_mask")
    output_dir = img_dir.parent / f"{img_dir.name}_fdesc"
    output_dir.mkdir(exist_ok=True)
    for filename in tqdm(img_dir.glob("*.png")):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        s = contours[0].reshape(-1, 2).astype(np.float32)
        keep = 16
        encoded = encode(s, keep=keep)
        t = decode(encoded)
        compress_ratio = s.size / (keep * 2)

        plt.figure(figsize=(8, 8))
        plt.plot(s[:, 0], s[:, 1], label="original", color="blue")
        plt.plot(t[:, 0], t[:, 1], label="reconstruct", color="red", linestyle="dashed")
        plt.legend()
        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"frdescp keep = {keep}, compression ratio = {compress_ratio}")
        save_path = output_dir / filename.name
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
