from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.models import TorchScriptBinaryClassifier
from src.utils import (
    IMG_EXTS, VID_EXTS,
    aggregate,
    clip_preprocess,
    iter_media,
    read_video_frames_uniform,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunked(tensor: torch.Tensor, chunk_size: int):
    for i in range(0, tensor.size(0), chunk_size):
        yield tensor[i:i + chunk_size]


@torch.no_grad()
def infer_image(
    clf: TorchScriptBinaryClassifier,
    image_path: Path,
    image_size: int,
    mean,
    std,
) -> float:
    img = Image.open(image_path).convert("RGB")
    x = clip_preprocess(img, image_size, mean, std).unsqueeze(0)  # (1,3,H,W)
    p_fake = clf.predict_fake_prob(x)[0].item()
    return float(p_fake)


@torch.no_grad()
def infer_video(
    clf: TorchScriptBinaryClassifier,
    video_path: Path,
    num_frames: int,
    agg_mode: str,
    frame_batch_size: int,
    image_size: int,
    mean,
    std,
) -> float | None:
    frames = read_video_frames_uniform(video_path, num_frames)
    if not frames:
        return None

    xs = [clip_preprocess(im, image_size, mean, std) for im in frames]   # list of (3,H,W)
    x = torch.stack(xs, dim=0)                                           # (T,3,H,W)

    probs = []
    for xb in chunked(x, max(1, frame_batch_size)):
        p_fake_b = clf.predict_fake_prob(xb).detach().cpu().tolist()
        probs.extend(p_fake_b)

    agg = aggregate([float(v) for v in probs], agg_mode)
    return None if agg is None else float(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    test_dir = Path(args.test_dir) if args.test_dir else Path(cfg["test_dir"])
    out_csv = Path(args.out_csv) if args.out_csv else Path(cfg["out_csv"])

    recursive = bool(cfg.get("recursive", True))
    num_frames = int(cfg.get("num_frames", 16))
    agg_mode = str(cfg.get("agg", "mean"))
    frame_batch_size = int(cfg.get("frame_batch_size", 16))

    image_size = int(cfg.get("image_size", 224))
    mean = cfg.get("clip_mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg.get("clip_std",  [0.26862954, 0.26130258, 0.27577711])

    model_pt = Path(cfg.get("model_pt", "model/model.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("test_dir:", str(test_dir))
    print("model_pt:", str(model_pt))

    if not model_pt.exists():
        raise FileNotFoundError(
            f"Missing model weights: {model_pt}\n"
        )

    clf = TorchScriptBinaryClassifier(model_pt=model_pt, device=device)

    media = sorted(list(iter_media(test_dir, recursive)))
    print(f"found {len(media)} media files")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = cfg.get("csv_header", ["filename", "label"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i, p in enumerate(media, 1):
            ext = p.suffix.lower()
            try:
                if ext in IMG_EXTS:
                    score = infer_image(clf, p, image_size, mean, std)
                elif ext in VID_EXTS:
                    score = infer_video(
                        clf, p,
                        num_frames=num_frames,
                        agg_mode=agg_mode,
                        frame_batch_size=frame_batch_size,
                        image_size=image_size,
                        mean=mean,
                        std=std,
                    )
                    if score is None:
                        print(f"[{i}/{len(media)}] {p.name}  SKIP(read fail)")
                        continue
                else:
                    continue

                fname = p.name

                w.writerow([fname, f"{score:.6f}"])
                print(f"[{i}/{len(media)}] {p.name}  fake_prob={score:.6f}")

            except Exception as e:
                print(f"[{i}/{len(media)}] {p.name}  ERROR: {e}")

    print("saved:", str(out_csv))


if __name__ == "__main__":
    main()
