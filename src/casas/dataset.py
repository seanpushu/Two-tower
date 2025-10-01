"""CASAS 事件流数据集。

提供训练采样和标签生成逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class QueryConfig:
    history_window_sec: int = 3600
    context_max_len: int = 256
    query_max_len: int = 32
    query_mode: str = "random"
    count_delta: int = 300
    query_span_sec: Optional[int] = None
    min_queries: int = 1
    binary_threshold: float = 0.0
    seed: Optional[int] = None
    value_key: str = "d_value"
    sensor_key: str = "d_name"
    timestamp_key: str = "ts"


class CasasDataset(Dataset):
    """CASAS 事件流抽样数据集。"""

    def __init__(
        self,
        events: Iterable,
        config: QueryConfig,
        split: str = "train",
        sensor_vocab: Optional[Dict[str, int]] = None,
    ):
        self.cfg = config
        self.split = split
        self._rng = np.random.default_rng(config.seed)

        (
            self.timestamps,
            self.sensor_indices,
            self.values,
            self.sensor_vocab,
            self.base_timestamp,
        ) = self._prepare_events(events, sensor_vocab)

        self.value_dim = self.values.shape[1]
        self.anchor_indices = self._build_anchor_indices()
        if len(self.anchor_indices) == 0:
            raise ValueError("没有可用的锚点，请检查窗口参数或数据长度。")

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, index):
        anchor_idx = self.anchor_indices[index]
        anchor_time = self.timestamps[anchor_idx]

        context_sensor, context_value, context_time = self._build_context(anchor_idx)
        query_time = self._build_queries(anchor_time)
        targets = self._build_targets(query_time)

        sample = {
            "anchor_time": torch.tensor(anchor_time, dtype=torch.float32),
            "context_sensor": torch.from_numpy(context_sensor.astype(np.int64)),
            "context_value": torch.from_numpy(context_value.astype(np.float32)),
            "context_time": torch.from_numpy(context_time.astype(np.float32)),
            "query_time": torch.from_numpy(query_time.astype(np.float32)),
        }
        sample.update({k: torch.from_numpy(v) for k, v in targets.items()})
        return sample

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    def _prepare_events(
        self,
        events: Iterable,
        sensor_vocab: Optional[Dict[str, int]],
    ):
        if isinstance(events, pd.DataFrame):
            df = events.copy()
        else:
            df = pd.DataFrame(events)

        cfg = self.cfg
        if cfg.timestamp_key not in df:
            raise KeyError(f"缺少时间戳列 {cfg.timestamp_key}")

        ts_raw = df[cfg.timestamp_key]
        if np.issubdtype(ts_raw.dtype, np.number):
            timestamps = ts_raw.astype(np.float64)
            base_ts = float(timestamps.min())
        else:
            ts_dt = pd.to_datetime(ts_raw)
            base_ts = ts_dt.min()
            timestamps = (ts_dt - base_ts).dt.total_seconds().astype(np.float64)

        df = df.assign(_ts=timestamps).sort_values("_ts").reset_index(drop=True)
        timestamps = df["_ts"].to_numpy(np.float64)

        if cfg.sensor_key not in df:
            df[cfg.sensor_key] = "sensor_0"

        if sensor_vocab is None:
            unique_sensors = sorted(df[cfg.sensor_key].astype(str).unique())
            sensor_vocab = {name: idx for idx, name in enumerate(unique_sensors)}

        sensor_indices = df[cfg.sensor_key].astype(str).map(sensor_vocab).to_numpy(np.int64)

        if cfg.value_key in df:
            values = df[cfg.value_key].to_numpy(np.float32)
        else:
            values = np.zeros(len(df), dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0).reshape(-1, 1)

        return timestamps, sensor_indices, values, sensor_vocab, base_ts

    def _build_anchor_indices(self):
        cfg = self.cfg
        history = cfg.history_window_sec
        if cfg.query_span_sec is not None:
            future_span = cfg.query_span_sec
        else:
            future_span = max(cfg.count_delta * cfg.query_max_len, cfg.count_delta)
        min_time = self.timestamps[0] + history
        max_time = self.timestamps[-1] - future_span

        anchors = []
        for idx, t in enumerate(self.timestamps):
            if t < min_time or t > max_time:
                continue
            left_bound = t - history
            left_idx = np.searchsorted(self.timestamps, left_bound, side="left")
            if idx - left_idx <= 0:
                continue
            anchors.append(idx)
        return anchors

    # ------------------------------------------------------------------
    # 构建上下文与查询
    # ------------------------------------------------------------------
    def _build_context(self, anchor_idx: int):
        cfg = self.cfg
        anchor_time = self.timestamps[anchor_idx]
        left_bound = anchor_time - cfg.history_window_sec
        left_idx = np.searchsorted(self.timestamps, left_bound, side="left")
        context_indices = slice(left_idx, anchor_idx)

        sensor = self.sensor_indices[context_indices]
        value = self.values[context_indices]
        time = self.timestamps[context_indices]

        if sensor.shape[0] > cfg.context_max_len:
            sensor = sensor[-cfg.context_max_len :]
            value = value[-cfg.context_max_len :]
            time = time[-cfg.context_max_len :]

        return sensor, value, time

    def _build_queries(self, anchor_time: float):
        cfg = self.cfg
        max_len = max(cfg.min_queries, cfg.query_max_len)
        if cfg.query_span_sec is not None:
            span = cfg.query_span_sec
        else:
            span = max(cfg.count_delta * cfg.query_max_len, cfg.count_delta)

        if self.split == "train" and cfg.query_mode == "random":
            num_queries = int(self._rng.integers(cfg.min_queries, max_len + 1))
            offsets = np.sort(self._rng.uniform(0.0, span, size=num_queries))
        else:
            num_queries = max_len
            step = span / max(1, num_queries)
            offsets = np.arange(1, num_queries + 1) * step

        query_time = anchor_time + offsets
        max_allowed = self.timestamps[-1]
        query_time = np.clip(query_time, None, max_allowed)
        return query_time.astype(np.float32)

    def _build_targets(self, query_time: np.ndarray) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        L = len(query_time)
        binary = np.zeros((L, 1), dtype=np.float32)
        count = np.zeros((L, 1), dtype=np.float32)
        tte = np.zeros((L, 1), dtype=np.float32)
        reg = np.zeros((L, self.value_dim), dtype=np.float32)

        mask_binary = np.zeros((L,), dtype=bool)
        mask_count = np.zeros((L,), dtype=bool)
        mask_tte = np.zeros((L,), dtype=bool)
        mask_reg = np.zeros((L,), dtype=bool)

        for i, qt in enumerate(query_time):
            # 最新观测值，用于二值与回归
            idx = np.searchsorted(self.timestamps, qt, side="right") - 1
            if idx >= 0:
                val = self.values[idx]
                reg[i] = val
                binary[i, 0] = 1.0 if val[0] > cfg.binary_threshold else 0.0
                mask_binary[i] = True
                mask_reg[i] = True

            # 计数标签
            end_time = qt + cfg.count_delta
            if end_time <= self.timestamps[-1]:
                start_idx = np.searchsorted(self.timestamps, qt, side="left")
                end_idx = np.searchsorted(self.timestamps, end_time, side="left")
                count[i, 0] = float(max(0, end_idx - start_idx))
                mask_count[i] = True

            # TTE 标签
            next_idx = np.searchsorted(self.timestamps, qt, side="right")
            if next_idx < len(self.timestamps):
                tte_value = self.timestamps[next_idx] - qt
                if tte_value > 0:
                    tte[i, 0] = np.log(tte_value)
                    mask_tte[i] = True

        targets = {
            "target_binary": binary,
            "target_count": count,
            "target_tte": tte,
            "target_reg": reg,
            "mask_binary": mask_binary.astype(np.bool_),
            "mask_count": mask_count.astype(np.bool_),
            "mask_tte": mask_tte.astype(np.bool_),
            "mask_reg": mask_reg.astype(np.bool_),
        }
        return targets


def casas_collate_fn(batch: List[Dict]):
    if len(batch) == 0:
        return {}

    def pad_sequence(tensors: List[torch.Tensor], pad_value=0.0):
        lengths = [t.size(0) for t in tensors]
        max_len = max(lengths)
        trailing_shape = tensors[0].shape[1:]
        dtype = tensors[0].dtype
        device = tensors[0].device
        output = torch.full((len(tensors), max_len, *trailing_shape), pad_value, dtype=dtype, device=device)
        mask = torch.ones((len(tensors), max_len), dtype=torch.bool, device=device)
        for i, tensor in enumerate(tensors):
            length = tensor.size(0)
            if length == 0:
                continue
            output[i, :length] = tensor
            mask[i, :length] = False
        return output, mask

    device = batch[0]["context_sensor"].device
    context_sensor, context_mask = pad_sequence([b["context_sensor"] for b in batch], pad_value=0)
    context_value, _ = pad_sequence([b["context_value"] for b in batch], pad_value=0.0)
    context_time, _ = pad_sequence([b["context_time"] for b in batch], pad_value=0.0)

    query_time, query_mask = pad_sequence([b["query_time"] for b in batch], pad_value=0.0)

    def pad_targets(key, mask_key, pad_value=0.0):
        tensors = [b[key] for b in batch]
        masks = [b[mask_key] for b in batch]
        lengths = [t.size(0) for t in tensors]
        max_len = max(lengths)
        trailing_shape = tensors[0].shape[1:]
        dtype = tensors[0].dtype
        output = torch.full((len(batch), max_len, *trailing_shape), pad_value, dtype=dtype, device=device)
        mask_out = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
        for i, (tensor, mask_tensor) in enumerate(zip(tensors, masks)):
            length = tensor.size(0)
            if length == 0:
                continue
            output[i, :length] = tensor.to(device)
            mask_out[i, :length] = mask_tensor.to(device)
        return output, mask_out

    target_binary, mask_binary = pad_targets("target_binary", "mask_binary")
    target_count, mask_count = pad_targets("target_count", "mask_count")
    target_tte, mask_tte = pad_targets("target_tte", "mask_tte")
    target_reg, mask_reg = pad_targets("target_reg", "mask_reg")

    anchor_time = torch.stack([b["anchor_time"] for b in batch]).to(device)
    loss_mask = mask_binary | mask_count | mask_tte | mask_reg

    return {
        "anchor_time": anchor_time,
        "context_sensor": context_sensor.to(device),
        "context_value": context_value.to(device),
        "context_time": context_time.to(device),
        "context_mask": context_mask.to(device),
        "query_time": query_time.to(device),
        "query_mask": query_mask.to(device),
        "target_binary": target_binary,
        "target_count": target_count,
        "target_tte": target_tte,
        "target_reg": target_reg,
        "mask_binary": mask_binary,
        "mask_count": mask_count,
        "mask_tte": mask_tte,
        "mask_reg": mask_reg,
        "loss_mask": loss_mask,
    }

