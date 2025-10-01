"""Two-Tower + Cross-Attention 模型实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .time_features import TimeFeatureConfig, TimeFeatureEncoder


@dataclass
class TwoTowerConfig:
    num_sensors: int
    d_model: int = 128
    value_dim: int = 1
    context_layers: int = 2
    context_heads: int = 4
    context_dropout: float = 0.1
    query_layers: int = 1
    query_heads: int = 4
    cross_layers: int = 2
    cross_heads: int = 4
    cross_dropout: float = 0.1
    binary_dim: int = 1
    count_dim: int = 1
    tte_dim: int = 1
    reg_dim: int = 1
    time_cfg: TimeFeatureConfig = None


class SensorEmbedding(nn.Module):
    def __init__(self, num_sensors: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_sensors, d_model)

    def forward(self, sensor_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(sensor_idx)


class ValueProjection(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.in_dim = in_dim
        if in_dim > 0:
            self.proj = nn.Linear(in_dim, d_model)
        else:
            self.register_parameter("proj", None)
            self.d_model = d_model

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.in_dim == 0 or self.proj is None:
            shape = values.shape[:-1] + (self.d_model,)
            return torch.zeros(shape, device=values.device, dtype=values.dtype)
        return self.proj(values)


class ContextTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        time_cfg = cfg.time_cfg or TimeFeatureConfig(d_model=cfg.d_model)
        self.time_encoder = TimeFeatureEncoder(time_cfg)
        self.sensor_embedding = SensorEmbedding(cfg.num_sensors, cfg.d_model)
        self.value_proj = ValueProjection(cfg.value_dim, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.context_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.context_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.context_layers)

    def forward(self, sensor_idx, values, timestamps, key_padding_mask=None):
        sensor_emb = self.sensor_embedding(sensor_idx)
        value_emb = self.value_proj(values)
        time_emb = self.time_encoder(timestamps)
        hidden = sensor_emb + value_emb + time_emb
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return hidden


class QueryTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        time_cfg = cfg.time_cfg or TimeFeatureConfig(d_model=cfg.d_model)
        self.time_encoder = TimeFeatureEncoder(time_cfg)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.query_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.context_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.query_layers)

    def forward(self, timestamps, key_padding_mask=None):
        hidden = self.time_encoder(timestamps)
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return hidden


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        query = query + self.dropout(attn_output)
        query = self.ln1(query)
        ffn_output = self.ffn(query)
        query = query + self.dropout(ffn_output)
        query = self.ln2(query)
        return query


class TwoTowerCrossAttentionModel(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.context_tower = ContextTower(cfg)
        self.query_tower = QueryTower(cfg)
        self.cross_layers = nn.ModuleList(
            [
                CrossAttentionBlock(cfg.d_model, cfg.cross_heads, cfg.cross_dropout)
                for _ in range(cfg.cross_layers)
            ]
        )

        self.head_bin = nn.Linear(cfg.d_model, cfg.binary_dim)
        self.head_cnt = nn.Linear(cfg.d_model, cfg.count_dim)
        self.head_tte = nn.Linear(cfg.d_model, cfg.tte_dim)
        self.head_reg = nn.Linear(cfg.d_model, cfg.reg_dim)

    def forward(self, batch):
        context = self.context_tower(
            batch["context_sensor"],
            batch["context_value"],
            batch["context_time"],
            key_padding_mask=batch.get("context_mask"),
        )
        query = self.query_tower(batch["query_time"], key_padding_mask=batch.get("query_mask"))

        Z = query
        for layer in self.cross_layers:
            Z = layer(Z, context, context, key_padding_mask=batch.get("context_mask"))

        logits_bin = self.head_bin(Z)
        rate_cnt = torch.nn.functional.softplus(self.head_cnt(Z)) + 1e-6
        log_tte = self.head_tte(Z)
        tte = torch.exp(log_tte)
        reg = self.head_reg(Z)

        return {
            "logits_bin": logits_bin,
            "rate_cnt": rate_cnt,
            "tte": tte,
            "log_tte": log_tte,
            "reg": reg,
        }

