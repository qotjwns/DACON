from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def get_full_frame_padded(pil_img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    img = pil_img.convert("RGB")
    img.thumbnail(target_size, Image.BICUBIC)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(
        img,
        ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2),
    )
    return new_img


def read_rgb_frames(
    file_path: Path,
    image_exts: Sequence[str],
    video_exts: Sequence[str],
    num_frames: int,
) -> List[np.ndarray]:
    ext = file_path.suffix.lower()

    if ext in set([e.lower() for e in image_exts]):
        try:
            img = Image.open(file_path).convert("RGB")
            return [np.array(img)]
        except Exception:
            return []

    if ext in set([e.lower() for e in video_exts]):
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        frame_indices = uniform_frame_indices(total, num_frames)
        frames: List[np.ndarray] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    return []


@dataclass
class PreprocessOutput:
    filename: str
    imgs: List[Image.Image]
    error: Optional[str] = None


def preprocess_one(
    file_path: Path,
    image_exts: Sequence[str],
    video_exts: Sequence[str],
    num_frames: int,
    target_size: Tuple[int, int],
    use_padding: bool = True,
) -> PreprocessOutput:
    try:
        frames = read_rgb_frames(
            file_path=file_path,
            image_exts=image_exts,
            video_exts=video_exts,
            num_frames=num_frames,
        )

        imgs: List[Image.Image] = []
        for rgb in frames:
            im = Image.fromarray(rgb)
            if use_padding:
                im = get_full_frame_padded(im, target_size)
            else:
                im = im.convert("RGB").resize(target_size, Image.BICUBIC)
            imgs.append(im)

        return PreprocessOutput(file_path.name, imgs, None)
    except Exception as e:
        return PreprocessOutput(file_path.name, [], str(e))
