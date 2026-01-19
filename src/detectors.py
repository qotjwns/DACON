from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import InferenceConfig
from src.face_preprocess import DnnFacePreprocessor


def build_face_preprocessor(cfg: InferenceConfig) -> Optional[DnnFacePreprocessor]:
    if not cfg.use_face_detector:
        return None
    if cfg.face_model_path is None or cfg.face_config_path is None:
        return None
    try:
        return DnnFacePreprocessor(
            model_path=cfg.face_model_path,
            config_path=cfg.face_config_path,
            image_size=cfg.image_size,
            mean=cfg.clip_mean,
            std=cfg.clip_std,
            margin_factor=cfg.face_margin,
            conf_threshold=cfg.face_conf_threshold,
            dump_dir=cfg.face_dump_dir,
        )
    except Exception:
        return None
