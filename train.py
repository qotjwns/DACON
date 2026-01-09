"""train.py

Baseline uses a pretrained Hugging Face ViT deepfake detector.
Training/fine-tuning is optional; implement here if your competition requires it.
"""

import argparse
from src.utils import load_config, seed_everything, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed")))
    device = resolve_device(cfg.get("device"))

    print(device)
    print("Loaded config OK.")
    print(f"Device resolved to: {device}")
    print("Baseline is inference-only by default.")


if __name__ == "__main__":
    main()
