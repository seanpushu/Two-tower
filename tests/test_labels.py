import os
import pathlib

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.append(str(SRC_DIR))

from casas.dataset import CasasDataset, QueryConfig, casas_collate_fn


def build_mock_events():
    timestamps = np.array([0, 60, 120, 180, 600, 1200, 1500], dtype=float)
    sensors = ["S1", "S2", "S1", "S1", "S2", "S1", "S2"]
    values = np.array([0, 1, 0, 1, 0, 2, 3], dtype=float)
    return pd.DataFrame({
        "ts": pd.to_datetime(timestamps, unit="s"),
        "d_name": sensors,
        "d_value": values,
    })


def test_labels_consistency():
    events = build_mock_events()
    cfg = QueryConfig(
        history_window_sec=300,
        context_max_len=10,
        query_max_len=4,
        count_delta=300,
        query_mode="grid",
        query_span_sec=600,
        binary_threshold=0.5,
        min_queries=2,
        seed=42,
    )

    dataset = CasasDataset(events, cfg, split="eval")
    sample = dataset[0]

    query_time = sample["query_time"].numpy()
    target_count = sample["target_count"].numpy().squeeze(-1)
    mask_count = sample["mask_count"].numpy()
    target_tte = sample["target_tte"].numpy().squeeze(-1)
    mask_tte = sample["mask_tte"].numpy()

    timestamps = dataset.timestamps

    for i, qt in enumerate(query_time):
        if not mask_count[i]:
            continue
        count_expected = ((timestamps >= qt) & (timestamps < qt + cfg.count_delta)).sum()
        assert target_count[i] == count_expected

    for i, qt in enumerate(query_time):
        if not mask_tte[i]:
            continue
        next_idx = np.searchsorted(timestamps, qt, side="right")
        assert next_idx < len(timestamps)
        tte_expected = np.log(timestamps[next_idx] - qt)
        np.testing.assert_allclose(target_tte[i], tte_expected)


def test_collate_padding():
    events = build_mock_events()
    cfg = QueryConfig(
        history_window_sec=300,
        context_max_len=10,
        query_max_len=3,
        count_delta=300,
        query_mode="grid",
        query_span_sec=600,
        binary_threshold=0.5,
        min_queries=2,
        seed=42,
    )

    dataset = CasasDataset(events, cfg, split="eval")
    batch = [dataset[0], dataset[1]]
    collated = casas_collate_fn(batch)

    assert collated["context_sensor"].shape[0] == 2
    assert collated["query_time"].shape[0] == 2
    assert collated["context_mask"].dtype == torch.bool
    assert collated["query_mask"].dtype == torch.bool

