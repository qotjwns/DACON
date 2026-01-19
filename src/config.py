from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Sequence

import yaml

DEFAULT_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
DEFAULT_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
VALID_AGG = {"mean", "median"}


def _as_path(val: str | Path | None, default: Path) -> Path:
    if val is None:
        return default
    return Path(val)


def load_yaml(path: Path | str) -> Mapping:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Config file must contain a mapping at the top level.")
    return data


@dataclass
class InferenceConfig:
    test_dir: Path
    out_csv: Path
    recursive: bool = True
    num_frames: int = 32
    agg: str = "mean"
    frame_batch_size: int = 16
    image_size: int = 224
    clip_mean: Sequence[float] = field(default_factory=lambda: list(DEFAULT_CLIP_MEAN))
    clip_std: Sequence[float] = field(default_factory=lambda: list(DEFAULT_CLIP_STD))
    model_pt: Path = Path("model/model.pt")
    csv_header: List[str] = field(default_factory=lambda: ["filename", "label"])
    device: str | None = None
    face_margin: float = 1.3
    face_dump_dir: Path | None = Path("output/faces")
    use_face_detector: bool = True
    face_model_path: Path | None = None   # OpenCV DNN weight path
    face_config_path: Path | None = None  # OpenCV DNN prototxt path
    face_conf_threshold: float = 0.5

    @classmethod
    def from_mapping(
        cls,
        data: Mapping,
        test_dir_override: str | Path | None = None,
        out_csv_override: str | Path | None = None,
    ) -> "InferenceConfig":
        return cls(
            test_dir=_as_path(test_dir_override, Path(data.get("test_dir", "test_data"))),
            out_csv=_as_path(out_csv_override, Path(data.get("out_csv", "submission.csv"))),
            recursive=bool(data.get("recursive", True)),
            num_frames=int(data.get("num_frames", 32)),
            agg=str(data.get("agg", "mean")),
            frame_batch_size=int(data.get("frame_batch_size", 16)),
            image_size=int(data.get("image_size", 224)),
            clip_mean=list(data.get("clip_mean", DEFAULT_CLIP_MEAN)),
            clip_std=list(data.get("clip_std", DEFAULT_CLIP_STD)),
            model_pt=_as_path(data.get("model_pt", "model/model.pt"), Path("model/model.pt")),
            csv_header=list(data.get("csv_header", ["filename", "label"])),
            device=str(data["device"]) if "device" in data else None,
            face_margin=float(data.get("face_margin", 1.3)),
            face_dump_dir=_as_path(data.get("face_dump_dir", "output/faces"), Path("output/faces"))
            if data.get("face_dump_dir", None) is not None
            else None,
            use_face_detector=bool(data.get("use_face_detector", True)),
            face_model_path=_as_path(data["face_model_path"], Path(".")) if "face_model_path" in data else None,
            face_config_path=_as_path(data["face_config_path"], Path(".")) if "face_config_path" in data else None,
            face_conf_threshold=float(data.get("face_conf_threshold", 0.5)),
        ).validated()

    def validated(self) -> "InferenceConfig":
        if self.agg not in VALID_AGG:
            raise ValueError(f"Invalid agg='{self.agg}'. Choose from {sorted(VALID_AGG)}.")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be > 0.")
        if self.frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be > 0.")
        if self.image_size <= 0:
            raise ValueError("image_size must be > 0.")
        if len(self.clip_mean) != 3 or len(self.clip_std) != 3:
            raise ValueError("clip_mean/clip_std must have length 3.")
        if not self.csv_header or len(self.csv_header) < 2:
            raise ValueError("csv_header must contain at least filename and label columns.")
        if self.face_margin <= 0:
            raise ValueError("face_margin must be > 0.")
        if self.use_face_detector and (self.face_model_path is None or self.face_config_path is None):
            raise ValueError("face_model_path and face_config_path must be set when use_face_detector is true.")
        return self

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"


def load_inference_config(
    path: str | Path,
    test_dir_override: str | Path | None = None,
    out_csv_override: str | Path | None = None,
) -> InferenceConfig:
    data = load_yaml(path)
    return InferenceConfig.from_mapping(data, test_dir_override, out_csv_override)
