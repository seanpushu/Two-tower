from typing import Dict

import torch
from sklearn.metrics import average_precision_score, f1_score


def compute_metrics(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
    metrics = {}
    logits_bin = preds.get("logits_bin")
    target_bin = targets.get("target_binary")
    mask_bin = targets.get("mask_binary")
    if logits_bin is not None and target_bin is not None:
        probs = logits_bin.sigmoid().detach().cpu().numpy()
        labels = target_bin.detach().cpu().numpy()
        if mask_bin is not None:
            mask = mask_bin.detach().cpu().numpy()
            probs = probs[mask]
            labels = labels[mask]
        if probs.size > 0:
            try:
                metrics["AUPRC"] = average_precision_score(labels, probs)
            except ValueError:
                metrics["AUPRC"] = float("nan")
            preds_bin = (probs > 0.5).astype(float)
            metrics["F1"] = f1_score(labels, preds_bin, zero_division=0)

    pred_reg = preds.get("reg")
    target_reg = targets.get("target_reg")
    mask_reg = targets.get("mask_reg")
    if pred_reg is not None and target_reg is not None:
        diff = (pred_reg - target_reg).detach()
        if mask_reg is not None:
            diff = diff.masked_select(mask_reg)
        if diff.numel() > 0:
            metrics["MAE"] = diff.abs().mean().item()
            metrics["RMSE"] = diff.pow(2).mean().sqrt().item()

    pred_tte = preds.get("log_tte")
    target_tte = targets.get("target_tte")
    mask_tte = targets.get("mask_tte")
    if pred_tte is not None and target_tte is not None:
        diff = (pred_tte - target_tte).detach()
        if mask_tte is not None:
            diff = diff.masked_select(mask_tte)
        if diff.numel() > 0:
            metrics["log-MAE"] = diff.abs().mean().item()

    return metrics

