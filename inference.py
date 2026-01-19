from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_inference_config
from src.face_preprocess import DnnFacePreprocessor
from src.models import TorchScriptBinaryClassifier
from src.pipeline import InferencePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = load_inference_config(
        args.config,
        test_dir_override=args.test_dir,
        out_csv_override=args.out_csv,
    )

    device = cfg.resolved_device()
    model_pt = Path(cfg.model_pt)

    print("device:", device)
    print("test_dir:", str(cfg.test_dir))
    print("model_pt:", str(model_pt))

    if not model_pt.exists():
        raise FileNotFoundError(f"Missing model weights: {model_pt}")

    face_preproc = None
    if cfg.use_face_detector:
        try:
            face_preproc = DnnFacePreprocessor(
                model_path=cfg.face_model_path,
                config_path=cfg.face_config_path,
                image_size=cfg.image_size,
                mean=cfg.clip_mean,
                std=cfg.clip_std,
                margin_factor=cfg.face_margin,
                conf_threshold=cfg.face_conf_threshold,
                dump_dir=cfg.face_dump_dir,
            )
        except Exception as e:
            print(f"WARNING: face detector init failed, fallback to center-crop. Error: {e}")
            face_preproc = None

    clf = TorchScriptBinaryClassifier(model_pt=model_pt, device=device)
    pipeline = InferencePipeline(cfg, clf, face_preprocessor=face_preproc)

    results = pipeline.run(show_progress=True)
    total = len(results)
    processed = sum(1 for r in results if r.score is not None)
    skipped = sum(1 for r in results if r.skipped)
    errors = [r for r in results if r.error]

    for idx, res in enumerate(results, 1):
        if res.error:
            print(f"[{idx}/{total}] {res.path.name}  ERROR: {res.error}")
        elif res.skipped:
            print(f"[{idx}/{total}] {res.path.name}  SKIP")
        else:
            print(f"[{idx}/{total}] {res.path.name}  fake_prob={res.score:.6f}")

    print(f"saved: {cfg.out_csv}")
    print(f"processed={processed}, skipped={skipped}, errors={len(errors)}")


if __name__ == "__main__":
    main()
