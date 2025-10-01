import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TimeFeatureConfig:
    d_model: int
    hour_sin_cos: bool = True
    weekday_sin_cos: bool = True
    use_rff: bool = True
    rff_dim: int = 16
    rff_scale: float = 1.0


class TimeFeatureEncoder(nn.Module):
    """时间特征编码器：支持周期 sin/cos 与随机傅里叶特征。"""

    def __init__(self, cfg: TimeFeatureConfig):
        super().__init__()
        self.cfg = cfg
        self.proj: Optional[nn.Linear]

        feature_dim = 0
        if cfg.hour_sin_cos:
            feature_dim += 2
        if cfg.weekday_sin_cos:
            feature_dim += 2
        if cfg.use_rff and cfg.rff_dim > 0:
            feature_dim += 2 * cfg.rff_dim
            # 预先生成频率
            self.register_buffer(
                "rff_freq",
                torch.randn(cfg.rff_dim) * cfg.rff_scale,
            )
        else:
            self.register_buffer("rff_freq", None)

        if feature_dim == 0:
            raise ValueError("时间特征维度为0，请至少启用一种时间特征")

        if feature_dim != cfg.d_model:
            self.proj = nn.Linear(feature_dim, cfg.d_model)
        else:
            self.proj = None

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """对时间戳张量进行编码。

        参数
        -----
        timestamps: torch.Tensor
            形状为 `(B, L)` 的时间戳，单位为秒或任意尺度。
        """

        cfg = self.cfg
        features = []
        device = timestamps.device

        # 转换为小时、星期等周期特征
        if cfg.hour_sin_cos or cfg.weekday_sin_cos:
            ts = timestamps.float()
            hours = (ts / 3600.0) % 24.0
            weeks = (ts / (3600.0 * 24.0)) % 7.0

            if cfg.hour_sin_cos:
                hour_angle = hours / 24.0 * 2 * math.pi
                hour_feat = torch.stack([torch.sin(hour_angle), torch.cos(hour_angle)], dim=-1)
                features.append(hour_feat)

            if cfg.weekday_sin_cos:
                week_angle = weeks / 7.0 * 2 * math.pi
                week_feat = torch.stack([torch.sin(week_angle), torch.cos(week_angle)], dim=-1)
                features.append(week_feat)

        if cfg.use_rff and self.rff_freq is not None:
            # RFF 使用正弦余弦对
            ts_scaled = timestamps.float().unsqueeze(-1) * self.rff_freq.to(device)
            rff_feat = torch.cat([torch.sin(ts_scaled), torch.cos(ts_scaled)], dim=-1)
            features.append(rff_feat)

        if not features:
            raise RuntimeError("未生成任何时间特征")

        features = torch.cat(features, dim=-1)

        if self.proj is not None:
            features = self.proj(features)
        return features

