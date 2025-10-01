from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    binary_weight: float = 1.0
    count_weight: float = 1.0
    tte_weight: float = 1.0
    reg_weight: float = 1.0
    use_uncertainty: bool = False


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_uncertainty:
            self.log_sigma = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ):
        device = next(iter(preds.values())).device
        losses = []
        weights = []

        logits_bin = preds.get("logits_bin")
        if logits_bin is not None and targets.get("target_binary") is not None:
            target = targets["target_binary"].to(device)
            mask = masks.get("mask_binary")
            loss_bin = F.binary_cross_entropy_with_logits(
                logits_bin, target, reduction="none"
            )
            if mask is not None:
                loss_bin = loss_bin.masked_fill(~mask.unsqueeze(-1), 0.0)
                denom = mask.unsqueeze(-1).float().sum().clamp_min(1.0)
            else:
                denom = torch.tensor(loss_bin.numel(), device=device)
            loss_bin = loss_bin.sum() / denom
            losses.append(loss_bin)
            weights.append(self.cfg.binary_weight)
        else:
            losses.append(torch.tensor(0.0, device=device))
            weights.append(0.0)

        rate_cnt = preds.get("rate_cnt")
        if rate_cnt is not None and targets.get("target_count") is not None:
            target = targets["target_count"].to(device)
            mask = masks.get("mask_count")
            loss_cnt = F.poisson_nll_loss(rate_cnt, target, log_input=False, reduction="none")
            if mask is not None:
                loss_cnt = loss_cnt.masked_fill(~mask.unsqueeze(-1), 0.0)
                denom = mask.unsqueeze(-1).float().sum().clamp_min(1.0)
            else:
                denom = torch.tensor(loss_cnt.numel(), device=device)
            loss_cnt = loss_cnt.sum() / denom
            losses.append(loss_cnt)
            weights.append(self.cfg.count_weight)
        else:
            losses.append(torch.tensor(0.0, device=device))
            weights.append(0.0)

        log_tte = preds.get("log_tte")
        if log_tte is not None and targets.get("target_tte") is not None:
            target = targets["target_tte"].to(device)
            mask = masks.get("mask_tte")
            diff = log_tte - target
            loss_tte = diff.pow(2)
            if mask is not None:
                loss_tte = loss_tte.masked_fill(~mask.unsqueeze(-1), 0.0)
                denom = mask.unsqueeze(-1).float().sum().clamp_min(1.0)
            else:
                denom = torch.tensor(loss_tte.numel(), device=device)
            loss_tte = loss_tte.sum() / denom
            losses.append(loss_tte)
            weights.append(self.cfg.tte_weight)
        else:
            losses.append(torch.tensor(0.0, device=device))
            weights.append(0.0)

        reg = preds.get("reg")
        if reg is not None and targets.get("target_reg") is not None:
            target = targets["target_reg"].to(device)
            mask = masks.get("mask_reg")
            loss_reg = torch.abs(reg - target)
            if mask is not None:
                loss_reg = loss_reg.masked_fill(~mask.unsqueeze(-1), 0.0)
                denom = mask.unsqueeze(-1).float().sum().clamp_min(1.0)
            else:
                denom = torch.tensor(loss_reg.numel(), device=device)
            loss_reg = loss_reg.sum() / denom
            losses.append(loss_reg)
            weights.append(self.cfg.reg_weight)
        else:
            losses.append(torch.tensor(0.0, device=device))
            weights.append(0.0)

        losses = torch.stack(losses)
        weights = torch.tensor(weights, device=device)

        if self.cfg.use_uncertainty:
            precision = torch.exp(-self.log_sigma)
            total_loss = (precision * losses * weights).sum() + self.log_sigma.sum()
        else:
            total_loss = (losses * weights).sum()

        return total_loss, losses

