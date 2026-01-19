from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

from src.config import InferenceConfig
from src.face_preprocess import FaceCropResult, DnnFacePreprocessor
from src.models import TorchScriptBinaryClassifier
from src.utils import (
    IMG_EXTS,
    VID_EXTS,
    aggregate,
    clip_preprocess,
    iter_media,
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
    ) -> Optional[torch.Tensor]:
        if self.face_preprocessor is not None and self.cfg.use_face_detector:
            face_res: Optional[FaceCropResult] = self.face_preprocessor.process(
                img,
                media_name=media_name,
                frame_idx=frame_idx,
            )
            if face_res is not None:
                return face_res.tensor
            # 얼굴 검출 실패 시 중앙 크롭으로 폴백
        return clip_preprocess(
            img,
            image_size=self.cfg.image_size,
            mean=self.cfg.clip_mean,
            std=self.cfg.clip_std,
        )

    def _infer_image(self, image_path: Path) -> Optional[float]:
        img = Image.open(image_path).convert("RGB")
        img_t = self._preprocess_image(img, media_name=image_path.stem)
        x = img_t.unsqueeze(0)
        return float(self.clf.predict_fake_prob(x)[0].item())

    def _infer_video(self, video_path: Path) -> Optional[float]:
        frames = read_video_frames_uniform(video_path, self.cfg.num_frames)
        if not frames:
            return None

        xs: List[torch.Tensor] = []
        for im, frame_idx in frames:
            x_t = self._preprocess_image(im, media_name=video_path.stem, frame_idx=frame_idx)
            if x_t is None:
                continue
            xs.append(x_t)

        if not xs:
            return None

        x = torch.stack(xs, dim=0)  # (T,3,H,W)

        probs: List[float] = []
        for xb in chunked_tensor(x, max(1, self.cfg.frame_batch_size)):
            p_fake_b = self.clf.predict_fake_prob(xb).detach().cpu().tolist()
            probs.extend(p_fake_b)

        agg = aggregate([float(v) for v in probs], self.cfg.agg)
        return None if agg is None else float(agg)

    def infer_path(self, path: Path) -> InferenceResult:
        ext = path.suffix.lower()
        try:
            if ext in IMG_EXTS:
                score = self._infer_image(path)
                if score is None:
                    return InferenceResult(path=path, score=None, skipped=True)
            elif ext in VID_EXTS:
                score = self._infer_video(path)
                if score is None:
                    return InferenceResult(path=path, score=None, skipped=True)
            else:
                return InferenceResult(path=path, score=None, skipped=True)
            return InferenceResult(path=path, score=score)
        except Exception as e:  # noqa: PERF203 acceptable for pipeline guardrail
            return InferenceResult(path=path, score=None, error=e)

    def iter_media(self) -> Iterable[Path]:
        yield from iter_media(self.cfg.test_dir, self.cfg.recursive)

    def run(self, show_progress: bool = True) -> List[InferenceResult]:
        media = sorted(list(self.iter_media()))
        results: List[InferenceResult] = []

        self.cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.cfg.csv_header)

            iterator = media
            if show_progress and tqdm is not None:
                iterator = tqdm(media, desc="infer", unit="file")

            for path in iterator:
                res = self.infer_path(path)
                if res.score is not None:
                    writer.writerow([path.name, f"{res.score:.6f}"])
                results.append(res)

        return results
