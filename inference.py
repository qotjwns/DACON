from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_inference_config
from src.detectors import build_face_preprocessor
from src.models import TorchScriptBinaryClassifier
from src.pipeline import InferencePipeline
from src.runner import InferenceRunner


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

    face_preproc = build_face_preprocessor(cfg)
    if face_preproc is None and cfg.use_face_detector:
        print("WARNING: face detector init failed or not configured; using center-crop fallback.")

    clf = TorchScriptBinaryClassifier(model_pt=model_pt, device=device)
    pipeline = InferencePipeline(cfg, clf, face_preprocessor=face_preproc)
    runner = InferenceRunner(cfg, pipeline)

    results, summary = runner.run(show_progress=True)
    total = len(results)

    for idx, res in enumerate(results, 1):
        if res.error:
            print(f"[{idx}/{total}] {res.path.name}  ERROR: {res.error}")
        elif res.skipped:
            print(f"[{idx}/{total}] {res.path.name}  SKIP")
        else:
            print(f"[{idx}/{total}] {res.path.name}  fake_prob={res.score:.6f}")

    print(f"saved: {cfg.out_csv}")
    print(
        f"processed={summary.processed}, skipped={summary.skipped}, "
        f"errors={summary.errors}, fallback_center_crop={summary.fallback}"
    )


if __name__ == "__main__":
    main()
