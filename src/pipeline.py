from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from src.config import InferenceConfig
from src.face_preprocess import FaceCropResult, DnnFacePreprocessor
from src.models import TorchScriptBinaryClassifier
from src.common.utils import (
    IMG_EXTS,
    VID_EXTS,
    aggregate,
    clip_preprocess,
    read_video_frames_uniform,
)


def chunked_tensor(tensor: torch.Tensor, chunk_size: int):
    for i in range(0, tensor.size(0), chunk_size):
        yield tensor[i : i + chunk_size]


@dataclass
class InferenceResult:
    path: Path
    score: Optional[float]
    error: Optional[Exception] = None
    skipped: bool = False
    fallback_used: bool = False


class InferencePipeline:
    def __init__(
        self,
        cfg: InferenceConfig,
        classifier: TorchScriptBinaryClassifier,
        face_preprocessor: Optional[DnnFacePreprocessor] = None,
    ):
        self.cfg = cfg
        self.clf = classifier
        self.face_preprocessor = face_preprocessor

    def _preprocess_image(
        self,
        img: Image.Image,
        media_name: str,
        frame_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, bool]:
        if self.face_preprocessor is not None and self.cfg.use_face_detector:
            face_res: Optional[FaceCropResult] = self.face_preprocessor.process(
                img,
                media_name=media_name,
                frame_idx=frame_idx,
            )
            if face_res is not None:
                return face_res.tensor, False
        tensor = clip_preprocess(
            img,
            image_size=self.cfg.image_size,
            mean=self.cfg.clip_mean,
            std=self.cfg.clip_std,
        )
        return tensor, bool(self.face_preprocessor and self.cfg.use_face_detector)

    def _infer_image(self, image_path: Path) -> Tuple[Optional[float], bool]:
        img = Image.open(image_path).convert("RGB")
        img_t, used_fallback = self._preprocess_image(img, media_name=image_path.stem)
        x = img_t.unsqueeze(0)
        score = float(self.clf.predict_fake_prob(x)[0].item())
        return score, used_fallback

    def _infer_video(self, video_path: Path) -> Tuple[Optional[float], bool]:
        frames = read_video_frames_uniform(video_path, self.cfg.num_frames)
        if not frames:
            return None, False

        xs: List[torch.Tensor] = []
        used_fallback = False
        for im, frame_idx in frames:
            x_t, fb = self._preprocess_image(im, media_name=video_path.stem, frame_idx=frame_idx)
            if x_t is None:
                continue
            xs.append(x_t)
            used_fallback = used_fallback or fb

        if not xs:
            return None, used_fallback

        x = torch.stack(xs, dim=0)  # (T,3,H,W)

        probs: List[float] = []
        for xb in chunked_tensor(x, max(1, self.cfg.frame_batch_size)):
            p_fake_b = self.clf.predict_fake_prob(xb).detach().cpu().tolist()
            probs.extend(p_fake_b)

        agg = aggregate([float(v) for v in probs], self.cfg.agg)
        return (None if agg is None else float(agg)), used_fallback

    def infer_path(self, path: Path) -> InferenceResult:
        ext = path.suffix.lower()
        try:
            if ext in IMG_EXTS:
                res_score, fb = self._infer_image(path)
                if res_score is None:
                    return InferenceResult(path=path, score=None, skipped=True)
                return InferenceResult(path=path, score=res_score, fallback_used=fb)
            if ext in VID_EXTS:
                res_score, fb = self._infer_video(path)
                if res_score is None:
                    return InferenceResult(path=path, score=None, skipped=True, fallback_used=fb)
                return InferenceResult(path=path, score=res_score, fallback_used=fb)
            return InferenceResult(path=path, score=None, skipped=True)
        except Exception as e:  # noqa: PERF203 acceptable for pipeline guardrail
            return InferenceResult(path=path, score=None, error=e)
