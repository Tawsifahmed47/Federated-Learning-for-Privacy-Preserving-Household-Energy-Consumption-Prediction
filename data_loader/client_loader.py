from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader.global_loader import GlobalDataLoader


class ClientDataLoader:
    def __init__(
        self,
        data_root: str | Path,
        target_column: str,
        batch_size: int,
        num_workers: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.clients_dir = self.data_root / "clients"
        self.target_column = target_column
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_client_datasets(
        self,
        global_loader: GlobalDataLoader,
        allowed_client_ids: List[str] | None = None,
    ) -> Tuple[Dict[str, DataLoader], List[int], int]:
        parquet_files = sorted(self.clients_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No client parquet files in {self.clients_dir}")

        allowed = set(allowed_client_ids) if allowed_client_ids is not None else None

        client_loaders: Dict[str, DataLoader] = {}
        sample_counts: List[int] = []
        input_dim = 0

        for file_path in parquet_files:
            client_id = file_path.stem
            if allowed is not None and client_id not in allowed:
                continue
            df = pd.read_parquet(file_path)
            x, y = global_loader.transform(df)
            input_dim = x.shape[1]
            sample_counts.append(len(df))
            dataset = TensorDataset(x, y)
            client_loaders[client_id] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        if not client_loaders:
            raise ValueError("No client datasets loaded. Check partition/client filters.")

        return client_loaders, sample_counts, input_dim

