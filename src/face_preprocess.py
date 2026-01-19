from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils import clip_preprocess


@dataclass
class FaceCropResult:
    tensor: torch.Tensor
    saved_path: Optional[Path] = None


class DnnFacePreprocessor:
    """
    OpenCV DNN 얼굴 검출기 사용:
    - 가장 큰 얼굴 박스를 선택
    - 마진 factor 적용 후 224x224 리사이즈 + CLIP 정규화
    - 검출 실패 시 None 반환 (파이프라인에서 중앙 크롭으로 폴백)
    """

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        margin_factor: float = 1.3,
        conf_threshold: float = 0.5,
        dump_dir: Optional[Path] = None,
    ):
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"DNN face model/config not found: {model_path} / {config_path}. "
                "config.yaml의 face_model_path/face_config_path를 확인하세요."
            )
        self.net = cv2.dnn.readNetFromCaffe(str(config_path), str(model_path))
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.margin_factor = margin_factor
        self.conf_threshold = conf_threshold
        self.dump_dir = dump_dir

    def _detect_largest(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0, size=(300, 300), mean=(104, 177, 123))
        self.net.setInput(blob)
        detections = self.net.forward()  # shape: (1,1,N,7)

        best = None
        best_area = -1.0
        if detections is None or detections.shape[2] == 0:
            return None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            x0 = int(detections[0, 0, i, 3] * w)
            y0 = int(detections[0, 0, i, 4] * h)
            x1 = int(detections[0, 0, i, 5] * w)
            y1 = int(detections[0, 0, i, 6] * h)
            area = max(0, x1 - x0) * max(0, y1 - y0)
            if area > best_area:
                best_area = area
                best = (x0, y0, x1, y1)
        return best

    def process(
        self,
        img: Image.Image,
        media_name: str,
        frame_idx: Optional[int] = None,
    ) -> Optional[FaceCropResult]:
        bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        det = self._detect_largest(bgr)
        if det is None:
            return None

        x0, y0, x1, y1 = det
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        w = (x1 - x0) * self.margin_factor
        h = (y1 - y0) * self.margin_factor

        nx0 = int(max(0, cx - w / 2))
        ny0 = int(max(0, cy - h / 2))
        nx1 = int(min(bgr.shape[1], cx + w / 2))
        ny1 = int(min(bgr.shape[0], cy + h / 2))
        if nx1 <= nx0 or ny1 <= ny0:
            return None

        face_crop = bgr[ny0:ny1, nx0:nx1, :]
        if face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_rgb).resize(
            (self.image_size, self.image_size), resample=Image.BICUBIC
        )
        tensor = clip_preprocess(face_img, self.image_size, self.mean, self.std)

        saved_path = None
        if self.dump_dir is not None:
            out_dir = Path(self.dump_dir) / media_name
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{frame_idx:05d}.png" if frame_idx is not None else "image.png"
            saved_path = out_dir / name
            face_img.save(saved_path, format="PNG", compress_level=0)

        return FaceCropResult(tensor=tensor, saved_path=saved_path)
