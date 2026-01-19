from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


def load_gend_class(repo_id: str):
    py_path = hf_hub_download(repo_id=repo_id, filename="modeling_gend.py")
    spec = importlib.util.spec_from_file_location("modeling_gend", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.GenD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="yermandy/GenD_CLIP_L_14")
    ap.add_argument("--out", default="model/model.vanilla.pt")
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    GenD = load_gend_class(args.repo)
    model = GenD.from_pretrained(args.repo).to(device).eval()

    # TorchScript: forward만 굳히기 (입력은 (B,3,224,224) 전처리 완료 텐서)
    example = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(out_path))

    print("saved torchscript:", str(out_path))


if __name__ == "__main__":
    main()
