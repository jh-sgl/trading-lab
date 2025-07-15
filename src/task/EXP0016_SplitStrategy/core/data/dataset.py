import logging
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import pandas as pd
import torch
from numba import njit
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ...external.alphasearch_naive.feature.ta_ops.ta_ops import add_ta
from ...util.const import RESAMPLE_RULE_TO_MIN, DFKey, Num
from ...util.registry import build_genfactor, register_dataset


@njit
def sharpe_to_n_min_numba(
    profits: np.ndarray, window_size: int, eps: float
) -> np.ndarray:
    n = len(profits)
    result = np.empty(n)
    for i in range(n):
        if window_size == -1:
            seg = profits[i:]
        else:
            seg = profits[i : i + window_size]

        if len(seg) < 2:
            result[i] = 0
        else:
            mean = seg.mean() * np.sqrt(252)
            var = np.mean((seg - mean) ** 2)
            std = np.sqrt(var)
            if std < 1e-8:
                result[i] = 0
            else:
                sharpe = mean / (std + eps)
                result[i] = sharpe
    return result


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


@register_dataset("basic_dataset")
class BasicDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        ta_factorset_fp: str,
        gen_factorset: list[DictConfig],
        train_date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        valid_date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        test_date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        repr_lookback_num: int,
        repr_lookahead_num: int,
        signal_stop_trade_after_n_min: int | None,
        signal_trade_between_hours: tuple[str, str] | None,
        signal_dayofweek: int | None,
        resample_rule: str,
        soft_label_hold_thresh: float,
        soft_label_tau: float,
        soft_label_mode: Literal["fixed", "dynamic"],
        signal_label_type: Literal["sharpe_at_market_close", "profit_at_market_close"],
    ) -> None:
        self._repr_lookback_num = repr_lookback_num
        self._repr_lookahead_num = repr_lookahead_num
        self._soft_label_hold_thresh = soft_label_hold_thresh
        self._soft_label_tau = soft_label_tau
        self._soft_label_mode = soft_label_mode
        self._resample_rule = resample_rule

        self._train_date_range = train_date_range
        self._valid_date_range = valid_date_range
        self._test_date_range = test_date_range
        self._total_date_range = (train_date_range[0], test_date_range[1])
        self._signal_dayofweek = signal_dayofweek

        df, self._factor_cols = self._load_data(
            data_fp,
            ta_factorset_fp,
            self._total_date_range,
            resample_rule,
            gen_factorset,
        )
        df, execution_price_col = self._set_execution_price(df)
        df, self._signal_label_col = self._add_signal_label_col(df, signal_label_type)
        df = self._drop_unused_cols(
            df,
            list(self._factor_cols)
            + [DFKey.DATE, self._signal_label_col, execution_price_col],
        )
        df = self._drop_unused_rows(df, self._factor_cols)
        self._df = self._scale_all_data_by_train_date_range(
            df, self._factor_cols, train_date_range
        )

        self._factor_tensor = torch.from_numpy(
            self._df[self._factor_cols].to_numpy(dtype=np.float32)
        )

        signal_labels = self._df[self._signal_label_col].to_numpy()
        if soft_label_mode == "fixed":
            signal_labels = np.stack(
                [self._create_soft_label(lbl) for lbl in signal_labels]
            )
            self._signal_label_tensor = torch.from_numpy(
                signal_labels.astype(np.float32)
            )
        elif soft_label_mode == "dynamic":
            self._signal_label_tensor = signal_labels

        self._timeindex_tensor = torch.from_numpy(self._df.index.values.astype("int64"))

        self._repr_tensor = None
        self._signal_train_repr_tensor = None
        self._dataloader_idx_to_repr_idx = None

        self._repr_train_dataloader_idx, self._repr_valid_dataloader_idx = (
            self._create_repr_dataloader_idx(
                self._df,
                self._train_date_range,
                self._valid_date_range,
            )
        )

        self._signal_train_dataloader_idx, self._signal_total_dataloader_idx = (
            self._create_signal_dataloader_idx(
                self._df,
                self._train_date_range,
                signal_stop_trade_after_n_min,
                signal_trade_between_hours,
                signal_dayofweek,
            )
        )

        self._learning_stage = "repr"

        logging.info(
            f"\nDataset setup for repr: df.shape={self._df.shape} / "
            f"# of train_sample={len(self._repr_train_dataloader_idx)} / "
            f"# of valid_sample={len(self._repr_valid_dataloader_idx)}\n"
            f"Dataset setup for signal: df.shape={self._df.shape} / "
            f"# of train_sample={len(self._signal_train_dataloader_idx)} / "
            f"# of total_sample={len(self._signal_total_dataloader_idx)}"
        )

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def factor_cols(self) -> pd.MultiIndex:
        return self._factor_cols

    @property
    def repr_train_dataloader_idx(self) -> list[int]:
        return self._repr_train_dataloader_idx

    @property
    def repr_valid_dataloader_idx(self) -> list[int]:
        return self._repr_valid_dataloader_idx

    @property
    def signal_train_dataloader_idx(self) -> list[int]:
        return self._signal_train_dataloader_idx

    @property
    def signal_total_dataloader_idx(self) -> list[int]:
        return self._signal_total_dataloader_idx

    @property
    def signal_train_repr_tensor(self) -> torch.Tensor:
        if self._signal_train_repr_tensor is None:
            selected_repr = []
            for didx in self._signal_train_dataloader_idx:
                ridx = self._dataloader_idx_to_repr_idx[didx]
                selected_repr.append(self._repr_tensor[ridx])
            self._signal_train_repr_tensor = torch.stack(selected_repr)
        return self._signal_train_repr_tensor

    @property
    def signal_train_label_tensor(self) -> torch.Tensor:
        if self._soft_label_mode == "fixed":
            label_tensor = self._signal_label_tensor[self._signal_train_dataloader_idx]
        elif self._soft_label_mode == "dynamic":
            label_tensor = torch.from_numpy(
                np.stack(
                    [
                        self._create_soft_label(self._signal_label_tensor[idx])
                        for idx in self._signal_train_dataloader_idx
                    ]
                ).astype(np.float32)
            )
        return label_tensor

    def _add_adj_option_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        opt_mask = df.columns.get_level_values(0).str.startswith("m0s_top")
        opt_df = df.loc[:, opt_mask].copy()
        opt_df.columns = ["__".join(col) for col in opt_df.columns.to_flat_index()]
        col_map = {col: i for i, col in enumerate(opt_df.columns)}

        for ohlc in ["OPEN", "HIGH", "LOW", "CLOSE"]:
            price = df[getattr(DFKey, f"FUTURE_PRICE_{ohlc}")].values

            call_strike = df[
                getattr(DFKey, f"M0S_TOP_TX_201_STRIKE_PRICE_{ohlc}")
            ].values
            put_strike = df[
                getattr(DFKey, f"M0S_TOP_TX_301_STRIKE_PRICE_{ohlc}")
            ].values

            call_gap = np.round((price - call_strike) / 2.5).astype(int)
            put_gap = np.round((price - put_strike) / 2.5).astype(int)

            n = len(df)
            offsets = np.arange(5)

            if (call_gap_over_20 := (abs(call_gap) > 20).sum()) > 0:
                logging.warning(
                    f"# of abs(call_gap) > 20 at {ohlc}: {call_gap_over_20} --- will be clipped to +-20"
                )
            if (put_gap_over_20 := (abs(put_gap) > 20).sum()) > 0:
                logging.warning(
                    f"# of abs(put_gap) > 20 at {ohlc}: {put_gap_over_20} --- will be clipped to +-20"
                )

            call_gaps = np.clip((call_gap[:, None] + offsets[None, :]), -20, 20)
            put_gaps = np.clip((put_gap[:, None] + offsets[None, :]), -20, 20)

            for i in range(5):
                cg = call_gaps[:, i]
                pg = put_gaps[:, i]

                for cat in ["price", "openint", "tx_amt", "tx_vol"]:
                    call_dfkey = f"ADJ_CALL_{i+1}_{cat.upper()}_{ohlc}"
                    put_dfkey = f"ADJ_PUT_{i+1}_{cat.upper()}_{ohlc}"

                    if cat.startswith("tx_"):
                        if ohlc == "CLOSE":
                            call_dfkey = "_".join(
                                call_dfkey.replace("TX_", "").split("_")[:-1]
                            )
                            put_dfkey = "_".join(
                                put_dfkey.replace("TX_", "").split("_")[:-1]
                            )
                        else:
                            continue

                    call_col_tgt = getattr(DFKey, call_dfkey)
                    put_col_tgt = getattr(DFKey, put_dfkey)

                    call_vals = np.empty(n, dtype=float)
                    put_vals = np.empty(n, dtype=float)

                    for idx in range(n):
                        call_col = f"m0s_top_tx_201_{cg[idx]:+d}_{cat}__{ohlc.lower()}"
                        put_col = f"m0s_top_tx_301_{pg[idx]:+d}_{cat}__{ohlc.lower()}"

                        if cat.startswith("tx_"):
                            call_col = f"m0s_top_tx_201_{cg[idx]:+d}_{cat}__volume"
                            put_col = f"m0s_top_tx_301_{pg[idx]:+d}_{cat}__volume"

                        call_vals[idx] = opt_df.values[idx, col_map[call_col]]
                        put_vals[idx] = opt_df.values[idx, col_map[put_col]]

                    df[call_col_tgt] = call_vals
                    df[put_col_tgt] = put_vals

        return df

    def _set_execution_price(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, tuple[str, str]]:
        df[DFKey.PRICE_EXECUTION] = (
            df.groupby(df.index.date)[[DFKey.FUTURE_PRICE_OPEN]].shift(-1).ffill()
        )
        return df, DFKey.PRICE_EXECUTION

    def set_learning_stage(
        self,
        learning_stage: Literal["repr", "signal", "signal_wo_repr"],
        repr_model: L.LightningModule | None = None,
    ) -> None:
        self._learning_stage = learning_stage

        if learning_stage == "signal" and repr_model is not None:
            with torch.no_grad():
                logging.info("Creating Representation for Signal Training...")
                reprs = []
                ft = self._factor_tensor.to(repr_model.device)
                for (
                    idx
                ) in self._signal_total_dataloader_idx:  # TODO: batchfy when needed
                    inputs = ft[idx - self._repr_lookback_num + 1 : idx + 1].unsqueeze(
                        0
                    )
                    reprs.append(repr_model(inputs))

                self._dataloader_idx_to_repr_idx = {
                    didx: ridx
                    for ridx, didx in enumerate(self._signal_total_dataloader_idx)
                }
                self._repr_tensor = torch.cat(reprs, dim=0).cpu().requires_grad_(False)

    def _create_repr_dataloader_idx(
        self,
        df: pd.DataFrame,
        train_date_range: tuple[str, str],
        valid_date_range: tuple[str, str],
    ) -> tuple[list[int], list[int]]:
        train_df = self._apply_date_range(df, train_date_range)
        valid_df = self._apply_date_range(df, valid_date_range)
        train_dataloader_idx = df.index.get_indexer(
            train_df.index[self._repr_lookback_num - 1 : -self._repr_lookahead_num]
        )
        valid_dataloader_idx = df.index.get_indexer(
            valid_df.index[self._repr_lookback_num - 1 : -self._repr_lookahead_num]
        )
        return train_dataloader_idx, valid_dataloader_idx

    def _create_signal_dataloader_idx(
        self,
        df: pd.DataFrame,
        train_date_range: tuple[str, str],
        stop_trade_after_n_min: int | None,
        signal_trade_between_hours: tuple[str, str] | None,
        signal_dayofweek: int | None,
    ) -> tuple[list[int], list[int]]:
        stop_trade_df = self._filter_trade_by_time(
            df, stop_trade_after_n_min, signal_trade_between_hours
        )
        stop_trade_train_df = self._apply_date_range(stop_trade_df, train_date_range)
        if signal_dayofweek is not None:
            stop_trade_train_df = stop_trade_train_df[
                stop_trade_train_df.index.dayofweek == signal_dayofweek
            ]
        train_dataloader_idx = df.index.get_indexer(
            stop_trade_train_df.index[self._repr_lookback_num - 1 :]
        )
        if signal_dayofweek is not None:
            stop_trade_df = stop_trade_df[
                stop_trade_df.index.dayofweek == signal_dayofweek
            ]
        total_dataloader_idx = df.index.get_indexer(
            stop_trade_df.index[self._repr_lookback_num - 1 :]
        )
        return train_dataloader_idx, total_dataloader_idx

    def _gumbel_noise(self, shape: tuple[int, ...]) -> np.ndarray:
        U = np.random.uniform(0, 1, shape)
        return -np.log(-np.log(U + Num.EPS) + Num.EPS)

    def _softmax(self, x: np.ndarray, tau: float) -> np.ndarray:
        exp_x = np.exp((x - np.max(x)) / tau)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def _create_soft_label(self, label: float) -> np.ndarray:
        pos_criterion = abs(label)

        if label > self._soft_label_hold_thresh:
            label_ = [0, self._soft_label_hold_thresh, pos_criterion]
        elif label < -self._soft_label_hold_thresh:
            label_ = [pos_criterion, self._soft_label_hold_thresh, 0]
        else:
            label_ = [0, self._soft_label_hold_thresh, 0]

        label = np.array(label_).astype(np.float32)
        label = label + self._gumbel_noise(label.shape)
        label = self._softmax(label, tau=self._soft_label_tau)
        return label

    def _filter_trade_by_time(
        self,
        df: pd.DataFrame,
        stop_trade_after_n_min: int | None,
        signal_trade_between_hours: tuple[str, str] | None,
    ) -> pd.DataFrame:
        if stop_trade_after_n_min is None and signal_trade_between_hours is None:
            return df

        mask = pd.Series(False, index=df.index)

        if signal_trade_between_hours is not None:
            start_hour, end_hour = signal_trade_between_hours
            mask |= (df.index.hour >= start_hour) & (df.index.hour <= end_hour)

        if stop_trade_after_n_min is not None:
            first_times = df.groupby(df.index.date).apply(lambda g: g.index.min())
            for _, first_time in first_times.items():
                start = first_time
                end = start + pd.Timedelta(minutes=stop_trade_after_n_min)
                mask |= (df.index >= start) & (df.index <= end)

        return df[mask].copy()

    def _add_signal_label_col(
        self,
        df: pd.DataFrame,
        signal_label_type: str,
    ) -> tuple[pd.DataFrame, tuple[str, str]]:
        if signal_label_type.startswith("sharpe_to"):
            profits = (
                df.groupby(df.index.date)[[DFKey.PRICE_EXECUTION]]
                .diff()
                .shift(-1)
                .fillna(0)
            )
            if np.isnan(profits.values).any():
                raise ValueError(f"Found nan in profits: {profits}")

            sharpe_to = signal_label_type.split("sharpe_to_")[-1]
            if sharpe_to == "eod":
                window_size = -1
            else:
                try:
                    window_size = max(
                        1, int(sharpe_to) // RESAMPLE_RULE_TO_MIN[self._resample_rule]
                    )
                except ValueError:
                    raise ValueError(f"Invalid signal_label_type: '{sharpe_to}'")

            df[DFKey.SHARPE_TO_X_MIN] = np.nan
            for _, idx in df.groupby(df.index.date).groups.items():
                daily_profit = profits.loc[idx].values.flatten()
                sharpe = sharpe_to_n_min_numba(daily_profit, window_size, Num.EPS)
                df.loc[idx, DFKey.SHARPE_TO_X_MIN] = sharpe
            return df, DFKey.SHARPE_TO_X_MIN

        elif signal_label_type == "profit_at_market_close":
            df[DFKey.PROFIT_AT_MARKET_CLOSE] = df.groupby(df.index.date)[
                [DFKey.FUTURE_PRICE_CLOSE]
            ].transform(lambda x: x.iloc[-1] - x)
            return df, DFKey.PROFIT_AT_MARKET_CLOSE

        else:
            raise ValueError(f"Invalid signal label type: {signal_label_type}")

    def _scale_all_data_by_train_date_range(
        self,
        df: pd.DataFrame,
        cols_to_scale: pd.MultiIndex,
        train_date_range: tuple[str, str],
    ) -> pd.DataFrame:
        train_df = self._apply_date_range(df, train_date_range)
        scaler = StandardScaler()
        scaler.fit(train_df[cols_to_scale].values)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale].values)
        return df

    def _drop_unused_rows(
        self, df: pd.DataFrame, subset_cols: pd.MultiIndex
    ) -> pd.DataFrame:
        n_rows_before = len(df)
        df = df.dropna(subset=subset_cols).copy()
        n_rows_after = len(df)
        logging.info(
            f"Dropped {n_rows_before - n_rows_after} rows during Dataset._drop_unused_rows()"
        )
        return df

    def _drop_unused_cols(
        self, df: pd.DataFrame, cols_to_keep: list[str | tuple[str, str]]
    ) -> pd.DataFrame:
        unused_cols = df.columns
        unused_cols = unused_cols.difference(cols_to_keep)

        logging.info(
            f"Dropped {len(unused_cols)} cols during Dataset._drop_unused_cols()"
        )
        return df.drop(columns=unused_cols).copy()

    def _apply_date_range(
        self, df: pd.DataFrame, date_range: tuple[str, str]
    ) -> pd.DataFrame:
        start, end = date_range
        time = df.index
        df = df[(start <= time) & (time <= end)].copy()
        return df

    def _load_data(
        self,
        data_fp: str | Path,
        ta_factorset_fp: str | None,
        total_date_range: tuple[str, str],
        resample_rule: str,
        gen_factorset: list[DictConfig],
    ) -> tuple[pd.DataFrame, pd.MultiIndex]:
        orig_data = pd.read_parquet(data_fp)

        data = orig_data[orig_data[DFKey.RESAMPLE_RULE] == resample_rule]
        data = self._apply_date_range(data, total_date_range)

        ta_cols = []
        if ta_factorset_fp is not None:
            factorset = torch.load(ta_factorset_fp, weights_only=False)
            data, ta_cols = add_ta(data, factorset)

        # data = self._add_adj_option_cols(data).copy()
        genfactor_cols = []
        gen_factorset = [
            build_genfactor(genfactor_cfg) for genfactor_cfg in gen_factorset
        ]
        for genfactor in gen_factorset:
            data, col = genfactor.add_genfactor(data)
            genfactor_cols.append(col)

        factor_cols = ta_cols + genfactor_cols
        if len(factor_cols) == 0:
            raise ValueError(
                "No input columns (both ta_factor and gen_factor are empty)"
            )
        return data, pd.MultiIndex.from_tuples(factor_cols)

    def _getitem_repr(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._factor_tensor[idx - self._repr_lookback_num + 1 : idx + 1]
        labels = self._factor_tensor[idx + 1 : idx + self._repr_lookahead_num + 1]
        timeindex = self._timeindex_tensor[idx]
        return inputs, labels, timeindex

    def _getitem_signal(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        repr_idx = self._dataloader_idx_to_repr_idx[idx]
        inputs = self._repr_tensor[repr_idx]

        if self._soft_label_mode == "fixed":
            labels = self._signal_label_tensor[idx]
        elif self._soft_label_mode == "dynamic":
            labels = self._create_soft_label(self._signal_label_tensor[idx])
        else:
            raise ValueError(f"Invalid soft label mode: {self._soft_label_mode}")

        timeindex = self._timeindex_tensor[idx]
        return inputs, labels, timeindex

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._learning_stage == "repr":
            return self._getitem_repr(idx)
        elif self._learning_stage == "signal":
            return self._getitem_signal(idx)
        else:
            raise ValueError(f"Invalid learning stage: {self._learning_stage}")

    def __len__(self) -> None:
        pass  # will be defined in Subset(); See datamodule.py
