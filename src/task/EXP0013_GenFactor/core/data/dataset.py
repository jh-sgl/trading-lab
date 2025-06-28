import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ...external.alphasearch_naive.feature.ta_ops.ta_ops import add_ta
from ...util.const import DFKey, Num
from ...util.registry import build_genfactor, register_dataset


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean


@register_dataset("basic")
class BasicDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        factorset_fp: str,
        gen_factorset: list[DictConfig] | None,
        gen_factorset_num: int,
        train_date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        total_date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        stop_trade_after_n_min: int,
        lookback_num: int,
        resample_rule: str,
        hold_thresh: float,
        tau: float,
    ) -> None:
        self._train_date_range = train_date_range
        self._total_date_range = total_date_range
        self._stop_trade_after_n_min = stop_trade_after_n_min
        self._hold_thresh = hold_thresh
        self._tau = tau

        df, self._factor_cols = self._load_data(
            data_fp,
            factorset_fp,
            total_date_range,
            resample_rule,
            gen_factorset,
            gen_factorset_num,
        )

        df, self._label_col = self._add_label_col(df)
        df = self._drop_unused_rows(df, self._factor_cols)
        df = self._drop_unused_cols(df, self._label_col, self._factor_cols)
        self._df, self._scaler = self._scale_data_by_train_date_range(
            df, self._factor_cols, train_date_range
        )
        self._train_dataloader_idx, self._total_dataloader_idx = (
            self._create_dataloader_idx(
                self._df,
                lookback_num,
                train_date_range,
                stop_trade_after_n_min,
            )
        )

        logging.info(
            f"Dataset created: df.shape={self._df.shape} / "
            f"# of train_sample={len(self._train_dataloader_idx)} / "
            f"# of total_sample={len(self._total_dataloader_idx)}"
        )

        self._input_tensor = torch.from_numpy(
            self._df[self._factor_cols].to_numpy(dtype=np.float32)
        )

        raw_labels = self._df[self._label_col].to_numpy()
        raw_labels = raw_labels / (raw_labels.std() + Num.EPS)
        soft_labels = np.stack([self._create_soft_label(lbl) for lbl in raw_labels])
        self._label_tensor = torch.from_numpy(soft_labels.astype(np.float32))

        self._timeindex_tensor = torch.from_numpy(self._df.index.values.astype("int64"))

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def factor_cols(self) -> pd.MultiIndex:
        return self._factor_cols

    @property
    def train_dataloader_idx(self) -> list[int]:
        return self._train_dataloader_idx

    @property
    def total_dataloader_idx(self) -> list[int]:
        return self._total_dataloader_idx

    @property
    def train_date_range(self) -> tuple[str, str]:
        return self._train_date_range

    @property
    def total_date_range(self) -> tuple[str, str]:
        return self._total_date_range

    def _add_label_col(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df[DFKey.PROFIT_AT_MARKET_CLOSE] = df.groupby(df.index.date)[
            [DFKey.FUTURE_PRICE_CLOSE]
        ].transform(lambda x: x.iloc[-1] - x)
        return df, DFKey.PROFIT_AT_MARKET_CLOSE

    def _scale_data_by_train_date_range(
        self,
        df: pd.DataFrame,
        cols_to_scale: pd.MultiIndex,
        train_date_range: tuple[str, str],
    ) -> tuple[pd.DataFrame, StandardScaler]:
        train_df = self._apply_date_range(df, train_date_range)
        scaler = StandardScaler()
        scaler.fit(train_df[cols_to_scale].values)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale].values)
        return df, scaler

    def _create_dataloader_idx(
        self,
        df: pd.DataFrame,
        lookback_num: int,
        train_date_range: tuple[str, str],
        stop_trade_after_n_min: int,
    ) -> tuple[list[int], list[int]]:
        stop_trade_df = self._apply_stop_trade_after_n_min(df, stop_trade_after_n_min)
        stop_trade_train_df = self._apply_date_range(stop_trade_df, train_date_range)
        train_dataloader_idx = df.index.get_indexer(
            stop_trade_train_df.index[lookback_num:]
        )
        total_dataloader_idx = df.index.get_indexer(stop_trade_df.index[lookback_num:])
        return train_dataloader_idx, total_dataloader_idx

    def _apply_stop_trade_after_n_min(
        self, df: pd.DataFrame, stop_trade_after_n_min: int
    ) -> pd.DataFrame:

        first_times = df.groupby(df.index.date).apply(lambda g: g.index.min())
        mask = pd.Series(False, index=df.index)
        for _, first_time in first_times.items():
            start = first_time
            end = start + pd.Timedelta(minutes=stop_trade_after_n_min)
            mask |= (df.index >= start) & (df.index <= end)
        return df[mask].copy()

    def _drop_unused_rows(
        self, df: pd.DataFrame, subset_cols: pd.MultiIndex
    ) -> pd.DataFrame:
        return df.dropna(subset=subset_cols).copy()

    def _drop_unused_cols(
        self, df: pd.DataFrame, label_col: tuple[str, str], factor_cols: pd.MultiIndex
    ) -> pd.DataFrame:
        cols_to_keep = (
            [label_col]
            + list(factor_cols)
            + [
                DFKey.FUTURE_PRICE_CLOSE,
                DFKey.FUTURE_PRICE_OPEN,
                DFKey.DATE,
            ]
        )
        return df.drop(columns=set(df.columns) - set(cols_to_keep)).copy()

    def _apply_date_range(
        self, df: pd.DataFrame, total_date_range: tuple[str, str]
    ) -> pd.DataFrame:
        start, end = total_date_range
        time = df.index
        df = df[(start <= time) & (time <= end)].copy()
        return df

    def _sample_genfactor_list(
        self,
        genfactor_list: list[DictConfig],
        k: int,
        use_existing_first: bool,
    ) -> list[DictConfig]:
        sampled = []
        if use_existing_first:
            sampled = genfactor_list[:k]
            if len(sampled) > k:
                random.shuffle(sampled)
                return sampled[:k]

        while len(sampled) < k:
            item = random.choice(genfactor_list)
            sampled.append(item)
        return sampled

    def _load_data(
        self,
        data_fp: str | Path,
        factorset_fp: str | None,
        total_date_range: tuple[str, str],
        resample_rule: str,
        gen_factorset: list[DictConfig],
        gen_factorset_num: int,
    ) -> tuple[pd.DataFrame, pd.MultiIndex]:
        orig_data = pd.read_parquet(data_fp)

        data = orig_data[orig_data[DFKey.RESAMPLE_RULE] == resample_rule]
        data = self._apply_date_range(data, total_date_range)

        ta_cols = []
        if factorset_fp is not None:
            factorset = torch.load(factorset_fp, weights_only=False)
            data, ta_cols = add_ta(data, factorset)

        genfactor_cols = []
        sampled_gen_factorset = self._sample_genfactor_list(
            gen_factorset, gen_factorset_num, use_existing_first=True
        )
        sampled_gen_factorset = [
            build_genfactor(genfactor_cfg) for genfactor_cfg in sampled_gen_factorset
        ]
        for genfactor in sampled_gen_factorset:
            data, col = genfactor.add_genfactor(data)
            genfactor_cols.append(col)

        factor_cols = ta_cols + genfactor_cols
        if len(factor_cols) == 0:
            raise ValueError(
                "No input columns (both ta_factor and gen_factor are empty)"
            )
        return data, pd.MultiIndex.from_tuples(factor_cols)

    def _prepare_inputs(self, idx: int) -> torch.Tensor:
        return self._input_tensor[idx]

    def _gumbel_noise(self, shape: tuple[int, ...]) -> np.ndarray:
        U = np.random.uniform(0, 1, shape)
        return -np.log(-np.log(U + Num.EPS) + Num.EPS)

    def _softmax(self, x: np.ndarray, tau: float) -> np.ndarray:
        exp_x = np.exp((x - np.max(x)) / tau)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def _create_soft_label(self, label: float) -> np.ndarray:
        pos_criterion = abs(label)

        if label > self._hold_thresh:
            label_ = [0, self._hold_thresh, pos_criterion]
        elif label < -self._hold_thresh:
            label_ = [pos_criterion, self._hold_thresh, 0]
        else:
            label_ = [0, self._hold_thresh, 0]

        label = np.array(label_).astype(np.float32)
        label = label + self._gumbel_noise(label.shape)
        label = self._softmax(label, tau=self._tau)
        return label

    def _prepare_label(self, idx: int) -> torch.Tensor:
        return self._label_tensor[idx]

    def _prepare_timeindex(self, idx: int) -> torch.Tensor:
        return self._timeindex_tensor[idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._prepare_inputs(idx)
        label = self._prepare_label(idx)
        timeindex = self._prepare_timeindex(idx)
        return inputs, label, timeindex

    def __len__(self) -> None:
        pass  # will be defined in Subset()
