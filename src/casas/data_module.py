"""CASAS 数据模块：负责加载 LS001 CSV 并构造数据集。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from torch.utils.data import DataLoader

from .dataset import CasasDataset, QueryConfig, casas_collate_fn


@dataclass
class CasasDataConfig:
    data_dir: str
    train_csv: str
    val_csv: str
    test_csv: str
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False


class CasasDataModule:
    def __init__(self, cfg: CasasDataConfig, query_cfg: QueryConfig):
        self.cfg = cfg
        self.query_cfg = query_cfg
        self.sensor_vocab: Optional[Dict[str, int]] = None

        self.train_dataset: Optional[CasasDataset] = None
        self.val_dataset: Optional[CasasDataset] = None
        self.test_dataset: Optional[CasasDataset] = None

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.cfg.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到数据文件: {path}")
        df = pd.read_csv(path)
        required_cols = {self.query_cfg.timestamp_key, self.query_cfg.sensor_key}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"CSV 缺少必要列: {missing}")
        return df

    def setup(self):
        train_df = self._load_csv(self.cfg.train_csv)
        self.train_dataset = CasasDataset(
            train_df,
            self.query_cfg,
            split="train",
            sensor_vocab=self.sensor_vocab,
        )
        self.sensor_vocab = self.train_dataset.sensor_vocab

        val_df = self._load_csv(self.cfg.val_csv)
        self.val_dataset = CasasDataset(
            val_df,
            self.query_cfg,
            split="eval",
            sensor_vocab=self.sensor_vocab,
        )

        test_df = self._load_csv(self.cfg.test_csv)
        self.test_dataset = CasasDataset(
            test_df,
            self.query_cfg,
            split="test",
            sensor_vocab=self.sensor_vocab,
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
            collate_fn=casas_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("数据集尚未初始化，请先调用 setup()")
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("数据集尚未初始化，请先调用 setup()")
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("数据集尚未初始化，请先调用 setup()")
        return self._make_loader(self.test_dataset, shuffle=False)

