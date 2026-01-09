from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.dataset import preprocess_one
from src.models import HFViTDeepfake
from src.utils import ensure_dir, load_config, resolve_device, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed")))

    device = resolve_device(cfg.get("device", "auto"))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]

    model = HFViTDeepfake(
        model_id=model_cfg["hf_model_id"],
        device=device,
        fake_class_index=int(model_cfg.get("fake_class_index")),
        cache_dir=model_cfg.get("cache_dir", None),
    )

    test_dir = Path(data_cfg["test_dir"])
    image_exts = [e.lower() for e in data_cfg.get("image_exts", [])]
    video_exts = [e.lower() for e in data_cfg.get("video_exts", [])]
    target_size = tuple(data_cfg.get("target_size"))
    num_frames = int(data_cfg.get("num_frames"))
    batch_size = int(data_cfg.get("batch_size"))
    use_padding = bool(data_cfg.get("use_padding"))
    print(num_frames)
    files = sorted([p for p in test_dir.iterdir() if p.is_file()])
    print(f"Device: {device}")
    print(f"Test data length: {len(files)}")
    print(f"Model: {model_cfg['hf_model_id']}")

    results: Dict[str, float] = {}

    for file_path in tqdm(files, desc="Processing"):
        out = preprocess_one(
            file_path=file_path,
            image_exts=image_exts,
            video_exts=video_exts,
            num_frames=num_frames,
            target_size=target_size,
            use_padding=use_padding,
        )

        if out.error:
            print(f"[WARN] {out.filename}: {out.error}")
            results[out.filename] = 0.0
            continue

        if not out.imgs:
            results[out.filename] = 0.0
            continue

        probs = model.predict_fake_probs(out.imgs, batch_size=batch_size)
        results[out.filename] = float(np.mean(probs)) if probs else 0.0

    sample_path = Path(data_cfg["sample_submission"])
    submission = pd.read_csv(sample_path)

    if "filename" not in submission.columns:
        raise ValueError("sample_submission.csv must have a 'filename' column.")
    if "prob" not in submission.columns:
        submission["prob"] = 0.0

    submission["prob"] = submission["filename"].map(results).fillna(0.0)

    out_dir = Path(out_cfg["out_dir"])
    ensure_dir(str(out_dir))
    out_csv = out_dir / out_cfg.get("out_csv", "baseline_submission.csv")
    submission.to_csv(out_csv, encoding="utf-8-sig", index=False)

    print(f"Inference completed. Processed: {len(results)} files")
    print(f"Saved submission to: {out_csv}")


if __name__ == "__main__":
    main()
    