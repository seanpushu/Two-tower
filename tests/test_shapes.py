import os
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.append(str(SRC_DIR))

from casas.model import TwoTowerConfig, TwoTowerCrossAttentionModel
from casas.time_features import TimeFeatureConfig


def test_model_output_shapes():
    torch.manual_seed(0)

    batch_size = 2
    context_lengths = [30, 50]
    query_lengths = [12, 7]
    context_max = max(context_lengths)
    query_max = max(query_lengths)

    num_sensors = 10
    d_model = 32
    value_dim = 1

    context_sensor = torch.zeros((batch_size, context_max), dtype=torch.long)
    context_value = torch.zeros((batch_size, context_max, value_dim))
    context_time = torch.zeros((batch_size, context_max))
    context_mask = torch.ones((batch_size, context_max), dtype=torch.bool)

    query_time = torch.zeros((batch_size, query_max))
    query_mask = torch.ones((batch_size, query_max), dtype=torch.bool)

    for b in range(batch_size):
        Lc = context_lengths[b]
        context_sensor[b, :Lc] = torch.randint(0, num_sensors, (Lc,))
        context_value[b, :Lc] = torch.randn(Lc, value_dim)
        context_time[b, :Lc] = torch.sort(torch.rand(Lc) * 1000)[0]
        context_mask[b, :Lc] = False

        Lq = query_lengths[b]
        query_time[b, :Lq] = torch.sort(torch.rand(Lq) * 1000 + 1000)[0]
        query_mask[b, :Lq] = False

    batch = {
        "context_sensor": context_sensor,
        "context_value": context_value,
        "context_time": context_time,
        "context_mask": context_mask,
        "query_time": query_time,
        "query_mask": query_mask,
    }

    time_cfg = TimeFeatureConfig(d_model=d_model, use_rff=True, rff_dim=8, rff_scale=0.01)
    config = TwoTowerConfig(
        num_sensors=num_sensors,
        d_model=d_model,
        value_dim=value_dim,
        binary_dim=2,
        count_dim=3,
        tte_dim=1,
        reg_dim=1,
        time_cfg=time_cfg,
    )

    model = TwoTowerCrossAttentionModel(config)
    outputs = model(batch)

    assert outputs["logits_bin"].shape == (batch_size, query_max, config.binary_dim)
    assert outputs["rate_cnt"].shape == (batch_size, query_max, config.count_dim)
    assert outputs["log_tte"].shape == (batch_size, query_max, config.tte_dim)
    assert outputs["tte"].shape == (batch_size, query_max, config.tte_dim)
    assert outputs["reg"].shape == (batch_size, query_max, config.reg_dim)
    assert torch.all(outputs["rate_cnt"] > 0)
    assert torch.all(outputs["tte"] > 0)

