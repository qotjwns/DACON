from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


class HFViTDeepfake:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        fake_class_index: int = 1,
        cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.fake_idx = int(fake_class_index)

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self.model = ViTForImageClassification.from_pretrained(model_id, **kwargs).to(device)
        self.processor = ViTImageProcessor.from_pretrained(model_id, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def predict_fake_probs(self, pil_images: List[Image.Image], batch_size: int = 32) -> List[float]:
        if not pil_images:
            return []

        probs: List[float] = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            p = F.softmax(logits, dim=1)[:, self.fake_idx]
            probs.extend(p.detach().cpu().tolist())
        return probs


if __name__ == "__main__":
    import argparse
    from torchinfo import summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="prithivMLmods/Deep-Fake-Detector-v2-Model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 모델 로드
    m = HFViTDeepfake(model_id=args.model_id, device=device)

    # 기본 정보 출력
    print("==== HF Model Info ====")
    print("model_id:", args.model_id)
    print("num_labels:", m.model.config.num_labels)