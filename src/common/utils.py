from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_media(root: str | Path, recursive: bool = True) -> Iterable[Path]:
    root_path = Path(root)
    paths = root_path.rglob("*") if recursive else root_path.glob("*")
    for p in paths:
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS or ext in VID_EXTS:
            yield p


def pick_frame_indices(total_frames: int, k: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= k:
        return list(range(total_frames))
    return [int(round(i * (total_frames - 1) / (k - 1))) for i in range(k)]


def aggregate(vals: Sequence[float], mode: str = "mean") -> Optional[float]:
    if not vals:
        return None
    if mode == "median":
        s = sorted(vals)
        m = len(s) // 2
        return s[m] if len(s) % 2 == 1 else (s[m - 1] + s[m]) / 2.0
    return float(sum(vals) / len(vals))


def _resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return img
    if w < h:
        new_w = size
        new_h = int(round(h * size / w))
    else:
        new_h = size
        new_w = int(round(w * size / h))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    right = min(w, left + size)
    bottom = min(h, top + size)
    return img.crop((left, top, right, bottom))


def clip_preprocess(
    img: Image.Image,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """
    CLIP style preprocessing: resize-shorter-side, center crop, normalize.
    """
    img = img.convert("RGB")
    img = _resize_shorter_side(img, image_size)
    img = _center_crop(img, image_size)

    arr = np.array(img).astype(np.float32) / 255.0  # (H,W,3)
    x = torch.from_numpy(arr).permute(2, 0, 1)      # (3,H,W)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x


def read_video_frames_uniform(
    video_path: Path,
    num_frames: int,
) -> List[Tuple[Image.Image, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = pick_frame_indices(total, num_frames) if total > 0 else []

    frames: List[Tuple[Image.Image, int]] = []

    if idxs:
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((Image.fromarray(rgb), fi))
    else:
        grabbed = 0
        fi = 0
        while grabbed < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((Image.fromarray(rgb), fi))
            grabbed += 1
            fi += 1

    cap.release()
    return frames
