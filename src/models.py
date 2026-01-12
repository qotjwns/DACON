from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class TorchScriptBinaryClassifier:
    """
    model/model.pt (TorchScript)를 로드해서 logits -> softmax 확률로 바꾸는 래퍼.
    기대 입력: float tensor (B,3,224,224) (CLIP normalize 적용된 것)
    기대 출력: logits (B,2)  where class0=REAL, class1=FAKE (원 코드 가정 유지)
    """

    def __init__(self, model_pt: str | Path, device: str):
        self.device = device
        self.model = torch.jit.load(str(model_pt), map_location=device)
        self.model.eval()

    @torch.no_grad()
    def predict_fake_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W)
        return: (B,) fake probability = 1 - P(real)
        """
        x = x.to(self.device, non_blocking=True)
        out = self.model(x)

        # TorchScript가 tuple/dict를 내는 경우 대비
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif isinstance(out, dict) and "logits" in out:
            logits = out["logits"]
        else:
            logits = out

        probs = torch.softmax(logits, dim=-1)   # (B,2)
        p_real = probs[:, 0]                    # class 0 = REAL
        p_fake = 1.0 - p_real
        return p_fake
