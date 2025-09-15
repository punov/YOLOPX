import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union

Tensor = torch.Tensor
SegOut = Dict[str, Tensor]  # {"da": logits[B,C,H,W], "ll": logits[B,C,H,W]}

def extract_seg_logits(out: Union[Dict, Tuple, list]) -> SegOut:
    """
    Extracting seg logits, stripping out dets if any
    Common patterns:
      - YOLOP/YOLOPX: model(x) -> det_out, da_logits, ll_logits
      - Custom dict: {"det":..., "da":..., "ll":...}
    """
    if isinstance(out, dict):
        return {"da": out["da"], "ll": out["ll"]}
    if isinstance(out, (tuple, list)):
        # Heuristic: last two are seg heads
        return {"da": out[-2], "ll": out[-1]}
    raise ValueError("Can't parse seg outputs from model forward()")

class PixelwiseKDLoss(nn.Module):
    """
    KL divergence on per-pixel class distributions with temperature T.
    For binary heads (C==1), we use BCE-with-logits toward teacher sigmoid.
    """
    def __init__(self, T: float = 4.0):
        super().__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, s_logits: Tensor, t_logits: Tensor) -> Tensor:
        if s_logits.shape[1] == 1:
            with torch.no_grad():
                t_prob = torch.sigmoid(t_logits.float())
            return F.binary_cross_entropy_with_logits(s_logits, t_prob)

        T = self.T
        s = F.log_softmax(s_logits.float() / T, dim=1)
        with torch.no_grad():
            t = F.softmax(t_logits.float() / T, dim=1)
        return self.kl(s, t) * (T * T)

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        C = logits.shape[1]
        if C == 1:
            prob = torch.sigmoid(logits)
            tgt = target.float().unsqueeze(1)
        else:
            prob = torch.softmax(logits, dim=1)
            tgt = F.one_hot(target.long().clamp_min(0), num_classes=C)
            tgt = tgt.permute(0, 3, 1, 2).float()
        inter = (prob * tgt).sum(dim=(2,3))
        den = prob.sum(dim=(2,3)) + tgt.sum(dim=(2,3)) + self.eps
        dice = 2 * inter / den
        return 1 - dice.mean()

class HardSegLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, use_dice: bool = True, dice_weight: float = 0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.use_dice = use_dice
        self.dice_w = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        C = logits.shape[1]
        if C == 1:
            # binary: target is {0,1}
            loss = self.bce(logits, target.float().unsqueeze(1))
        else:
            loss = self.ce(logits, target.long())
        if self.use_dice:
            loss = loss + self.dice_w * self.dice(logits, target)
        return loss

class SegDistiller(nn.Module):
    """
    Combines hard loss vs GT and KD loss vs teacher per head.
    total = alpha * (hard_da + hard_ll) + beta * (kd_da + kd_ll)
    """
    def __init__(self, teacher: nn.Module, student: nn.Module, T: float = 4.0,
                 alpha: float = 0.5, beta: float = 0.5,
                 ignore_index: int = 255, use_dice: bool = True, dice_weight: float = 0.2):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = student
        self.hard = HardSegLoss(ignore_index, use_dice, dice_weight)
        self.kd = PixelwiseKDLoss(T)
        self.alpha, self.beta = alpha, beta

    def forward(self, images: Tensor, targets: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        with torch.no_grad():
            t_out = extract_seg_logits(self.teacher(images))
        s_out = extract_seg_logits(self.student(images))

        hard_da = self.hard(s_out["da"], targets["da"])
        hard_ll = self.hard(s_out["ll"], targets["ll"])
        kd_da = self.kd(s_out["da"], t_out["da"])
        kd_ll = self.kd(s_out["ll"], t_out["ll"])

        total = self.alpha * (hard_da + hard_ll) + self.beta * (kd_da + kd_ll)
        logs = {
            "total": float(total.detach().cpu()),
            "hard_da": float(hard_da.detach().cpu()),
            "hard_ll": float(hard_ll.detach().cpu()),
            "kd_da": float(kd_da.detach().cpu()),
            "kd_ll": float(kd_ll.detach().cpu()),
        }
        return total, logs
