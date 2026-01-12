from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .utils import iter_media


@dataclass
class MediaItem:
    path: Path


def build_media_list(root: str | Path, recursive: bool = True) -> List[MediaItem]:
    return [MediaItem(p) for p in iter_media(root, recursive)]
