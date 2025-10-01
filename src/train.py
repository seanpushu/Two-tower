"""CASAS Two-Tower 模型训练脚本。"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from casas import (
    CasasDataConfig,
    CasasDataModule,
    LossConfig,
    MultiTaskLoss,
    QueryConfig,
    TimeFeatureConfig,
    TwoTowerConfig,
    TwoTowerCrossAttentionModel,
    compute_metrics,
)
from config import cfg, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Two-Tower model on CASAS data")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yml")))
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    return parser.parse_args()


def build_optimizer(model: nn.Module, optimizer_cfg: Dict) -> torch.optim.Optimizer:
    name = optimizer_cfg.get("name", "Adam").lower()
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, scheduler_cfg: Dict):
    name = scheduler_cfg.get("name", "none").lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=scheduler_cfg.get("t_max", 100))
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    return None


def save_checkpoint(state: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def main():
    args = parse_args()
    config_path = args.config
    user_cfg = load_config(config_path)

    # 覆盖全局 cfg
    cfg.clear()
    cfg.update(user_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device", "cuda") == "cuda" else "cpu")
    torch.manual_seed(cfg.get("init_seed", 0))

    dataset_cfg = cfg["dataset"]
    data_config = CasasDataConfig(
        data_dir=dataset_cfg["data_dir"],
        train_csv=dataset_cfg["train_csv"],
        val_csv=dataset_cfg.get("val_csv", dataset_cfg["train_csv"]),
        test_csv=dataset_cfg.get("test_csv", dataset_cfg["val_csv"]),
        batch_size=cfg["control"]["batch_size"],
        num_workers=cfg["control"].get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", True),
    )

    query_config = QueryConfig(
        history_window_sec=dataset_cfg["history_window_sec"],
        context_max_len=dataset_cfg["context_max_len"],
        query_max_len=dataset_cfg["query_max_len"],
        query_mode=dataset_cfg.get("query_mode", "random"),
        count_delta=dataset_cfg.get("count_delta", 300),
        query_span_sec=dataset_cfg.get("query_span_sec"),
        binary_threshold=dataset_cfg.get("binary_threshold", 0.5),
        min_queries=dataset_cfg.get("min_queries", 1),
        seed=cfg.get("init_seed", 0),
    )

    data_module = CasasDataModule(data_config, query_config)
    data_module.setup()

    time_cfg = cfg.get("time_features", {})
    time_config = TimeFeatureConfig(
        d_model=cfg["model"]["d_model"],
        hour_sin_cos=time_cfg.get("hour_sin_cos", True),
        weekday_sin_cos=time_cfg.get("weekday_sin_cos", True),
        use_rff=time_cfg.get("use_rff", True),
        rff_dim=time_cfg.get("rff_dim", 16),
        rff_scale=time_cfg.get("rff_scale", 0.01),
    )

    model_config = TwoTowerConfig(
        num_sensors=cfg["model"]["num_sensors"],
        d_model=cfg["model"]["d_model"],
        value_dim=cfg["model"].get("value_dim", 1),
        context_layers=cfg["model"].get("context_layers", 2),
        context_heads=cfg["model"].get("context_heads", 4),
        query_layers=cfg["model"].get("query_layers", 1),
        query_heads=cfg["model"].get("query_heads", 4),
        cross_layers=cfg["model"].get("cross_layers", 2),
        cross_heads=cfg["model"].get("cross_heads", 4),
        binary_dim=cfg["model"].get("binary_dim", 1),
        count_dim=cfg["model"].get("count_dim", 1),
        tte_dim=cfg["model"].get("tte_dim", 1),
        reg_dim=cfg["model"].get("reg_dim", 1),
        time_cfg=time_config,
    )

    model = TwoTowerCrossAttentionModel(model_config).to(device)

    loss_cfg = cfg.get("loss", {})
    loss_config = LossConfig(
        binary_weight=loss_cfg.get("binary_weight", 1.0),
        count_weight=loss_cfg.get("count_weight", 1.0),
        tte_weight=loss_cfg.get("tte_weight", 1.0),
        reg_weight=loss_cfg.get("reg_weight", 1.0),
        use_uncertainty=loss_cfg.get("use_uncertainty", False),
    )
    criterion = MultiTaskLoss(loss_config)

    optimizer = build_optimizer(model, cfg.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}))

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    checkpoint_dir = cfg["checkpoint"]["dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    best_path = os.path.join(checkpoint_dir, "best.pt")
    start_epoch = 0
    best_metric = float("-inf")

    if args.resume and os.path.exists(ckpt_path):
        state = load_checkpoint(ckpt_path, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state.get("epoch", 0)
        best_metric = state.get("best_metric", best_metric)

    writer = None
    if cfg["train"].get("tensorboard", True):
        writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "runs"))

    epochs = cfg["train"].get("epochs", 10)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    log_interval = cfg["train"].get("log_interval", 10)

    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad()
            outputs = model(batch)
            preds = {
                "logits_bin": outputs["logits_bin"],
                "rate_cnt": outputs["rate_cnt"],
                "log_tte": outputs["log_tte"],
                "reg": outputs["reg"],
            }
            targets = {
                "target_binary": batch["target_binary"],
                "target_count": batch["target_count"],
                "target_tte": batch["target_tte"],
                "target_reg": batch["target_reg"],
            }
            masks = {
                "mask_binary": batch["mask_binary"],
                "mask_count": batch["mask_count"],
                "mask_tte": batch["mask_tte"],
                "mask_reg": batch["mask_reg"],
            }
            loss, loss_items = criterion(preds, targets, masks)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()
            if writer:
                writer.add_scalar("train/loss", loss.item(), global_step)
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch+1}/{epochs} Step {batch_idx+1}/{len(train_loader)} "
                    f"Loss: {running_loss / (batch_idx + 1):.4f} Time/step: {elapsed / (batch_idx + 1):.3f}s"
                )
            global_step += 1

        val_metrics = evaluate(model, criterion, val_loader, device)
        if writer:
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        metric_to_track = val_metrics.get("AUPRC", val_metrics.get("MAE", 0.0))
        if metric_to_track > best_metric:
            best_metric = metric_to_track
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch + 1,
                    "best_metric": best_metric,
                },
                best_path,
            )
            print(f"New best metric {best_metric:.4f}, model saved to {best_path}")

        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch + 1,
                "best_metric": best_metric,
            },
            ckpt_path,
        )
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(-metric_to_track)
            else:
                scheduler.step()

    if writer:
        writer.close()


def evaluate(model, criterion, dataloader, device):
    model.eval()
    metrics_sum = {}
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(batch)
            preds = {
                "logits_bin": outputs["logits_bin"],
                "reg": outputs["reg"],
                "log_tte": outputs["log_tte"],
            }
            targets = {
                "target_binary": batch["target_binary"],
                "target_reg": batch["target_reg"],
                "target_tte": batch["target_tte"],
                "mask_binary": batch["mask_binary"],
                "mask_reg": batch["mask_reg"],
                "mask_tte": batch["mask_tte"],
            }
            metric_values = compute_metrics(preds, targets)
            for k, v in metric_values.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            total_batches += 1

    return {k: v / max(total_batches, 1) for k, v in metrics_sum.items()}


if __name__ == "__main__":
    main()

