from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from src.config import InferenceConfig
from src.pipeline import InferencePipeline, InferenceResult
from src.common.utils import iter_media


@dataclass
class RunSummary:
    processed: int
    skipped: int
    errors: int
    fallback: int


class InferenceRunner:
    def __init__(self, cfg: InferenceConfig, pipeline: InferencePipeline):
        self.cfg = cfg
        self.pipeline = pipeline

    def iter_media(self) -> Iterable[Path]:
        yield from iter_media(self.cfg.test_dir, self.cfg.recursive)

    def run(self, show_progress: bool = True) -> Tuple[List[InferenceResult], RunSummary]:
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
                res = self.pipeline.infer_path(path)
                if res.score is not None:
                    writer.writerow([path.name, f"{res.score:.6f}"])
                results.append(res)

        summary = RunSummary(
            processed=sum(1 for r in results if r.score is not None),
            skipped=sum(1 for r in results if r.skipped),
            errors=sum(1 for r in results if r.error),
            fallback=sum(1 for r in results if r.fallback_used),
        )
        return results, summary
