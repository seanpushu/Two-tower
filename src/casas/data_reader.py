"""CASAS 数据读取工具。"""

from __future__ import annotations

import pandas as pd


def load_csv(path: str):
    return pd.read_csv(path)

