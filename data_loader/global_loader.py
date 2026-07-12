from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class GlobalDataLoader:
    def __init__(
        self,
        data_root: str | Path,
        target_column: str,
        batch_size: int,
        drop_columns: List[str] | None = None,
        use_categorical: bool = False,
        max_categorical_levels: int = 50,
        num_workers: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.target_column = target_column
        self.batch_size = batch_size
        self.drop_columns = set(drop_columns or [])
        self.use_categorical = use_categorical
        self.max_categorical_levels = max_categorical_levels
        self.num_workers = num_workers
        self.preprocessor: Pipeline | None = None
        self.feature_names_: List[str] = []

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")
        x_df = df.drop(columns=[self.target_column])
        removable = [c for c in self.drop_columns if c in x_df.columns]
        if removable:
            x_df = x_df.drop(columns=removable)
        y = df[self.target_column].astype(float)
        return x_df, y

    def _build_preprocessor(self, x_df: pd.DataFrame) -> Pipeline:
        num_cols = x_df.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = [c for c in x_df.columns if c not in num_cols]

        # Guardrail: remove very high-cardinality categorical columns that cause
        # one-hot dimensional explosion and huge dense allocations.
        safe_cat_cols: List[str] = []
        if self.use_categorical:
            for c in cat_cols:
                n_unique = int(x_df[c].nunique(dropna=True))
                if n_unique <= self.max_categorical_levels:
                    safe_cat_cols.append(c)

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers = [("num", num_pipeline, num_cols)]
        if safe_cat_cols:
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            dtype="float32",
                        ),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipeline, safe_cat_cols))

        transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        return Pipeline(steps=[("transform", transformer)])

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        x_df, y = self._prepare_xy(train_df)
        self.preprocessor = self._build_preprocessor(x_df)
        x_np = self.preprocessor.fit_transform(x_df)
        self.feature_names_ = [f"f_{i}" for i in range(x_np.shape[1])]
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        return x_tensor, y_tensor

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        x_df, y = self._prepare_xy(df)
        x_np = self.preprocessor.transform(x_df)
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        return x_tensor, y_tensor

    def load_centralized(self) -> Tuple[DataLoader, DataLoader, int]:
        train_df = pd.read_parquet(self.data_root / "train.parquet")
        test_df = pd.read_parquet(self.data_root / "test.parquet")

        x_train, y_train = self.fit_transform(train_df)
        x_test, y_test = self.transform(test_df)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, test_loader, x_train.shape[1]

