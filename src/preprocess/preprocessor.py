from pathlib import Path
from typing import Literal

import pandas as pd
import ray
from ray.util import ActorPool
from tqdm import tqdm

from preprocess.raw_preprocessing_module import RawPreprocessingModuleBase
from preprocess.resample_preprocessing_module import ResamplePreprocessingModuleBase

ResampleRule = Literal["1min", "5min", "15min", "1h"]


class BasicPreprocessor:

    @ray.remote
    class Actor:
        def __init__(
            self,
            raw_preprocessing_pipeline: list[RawPreprocessingModuleBase],
            resample_rules: list[ResampleRule],
            resample_preprocessing_pipeline: list[ResamplePreprocessingModuleBase],
        ) -> None:
            self._raw_preprocessing_pipeline = raw_preprocessing_pipeline
            self._resample_rules = resample_rules
            self._resample_preprocessing_pipeline = resample_preprocessing_pipeline

        def _read_hdf_with_path(self, hdf_fp: Path) -> pd.DataFrame:
            orig_df = pd.read_hdf(hdf_fp)
            orig_df.fp = hdf_fp
            return orig_df

        def process(
            self,
            hdf_fp: Path,
        ) -> pd.DataFrame:
            raw_df = self._read_hdf_with_path(hdf_fp)

            raw_preprocessed_df = self._run_raw_preprocessing_pipeline(raw_df)
            resample_preprocessed_df = self._run_resample_preprocessing_pipeline(raw_preprocessed_df)
            final_df = self._flatten_columns(resample_preprocessed_df)

            return final_df

        def _run_raw_preprocessing_pipeline(self, raw_df: pd.DataFrame) -> pd.DataFrame:
            raw_preprocessed_df = raw_df
            for preprocessing_module in self._raw_preprocessing_pipeline:
                raw_preprocessed_df = preprocessing_module(raw_preprocessed_df)
            return raw_preprocessed_df

        def _run_resample_preprocessing_pipeline(
            self,
            raw_preprocessed_df: pd.DataFrame,
        ) -> pd.DataFrame:
            result_dfs = []
            for resample_rule in self._resample_rules:
                result_df = pd.DataFrame()
                resampled_df = raw_preprocessed_df.resample(resample_rule)

                for preprocessing_module in self._resample_preprocessing_pipeline:
                    result_df = preprocessing_module(result_df, resampled_df)
                result_df["resample_rule"] = resample_rule
                result_dfs.append(result_df)

            resample_preprocessed_df = pd.concat(result_dfs)
            return resample_preprocessed_df

        def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
            df.columns = ["_".join(col) if isinstance(col, tuple) else col for col in df.columns]
            return df

    def __init__(
        self,
        raw_preprocessing_pipeline: list[RawPreprocessingModuleBase],
        resample_rules: list[ResampleRule],
        resample_preprocessing_pipeline: list[ResamplePreprocessingModuleBase],
        raw_data_dir: str,
        ray_actor_num: int,
        save_fp: str,
    ) -> None:
        super().__init__()
        self._raw_data_fp_list = list(Path(raw_data_dir).glob("./*.h5"))
        self._save_fp = Path(save_fp)
        self._resample_rules = resample_rules
        self._ray_actor_num = ray_actor_num
        self._raw_preprocessing_pipeline = raw_preprocessing_pipeline
        self._resample_preprocessing_pipeline = resample_preprocessing_pipeline

    def _process_using_ray(self) -> list[pd.DataFrame]:
        processed_list = []

        actor_pool = ActorPool(
            [
                self.Actor.remote(
                    self._raw_preprocessing_pipeline,
                    self._resample_rules,
                    self._resample_preprocessing_pipeline,
                )
                for _ in range(self._ray_actor_num)
            ]
        )

        for result in tqdm(
            actor_pool.map(
                lambda a, fp: a.process.remote(fp),
                self._raw_data_fp_list,
            ),
            total=len(self._raw_data_fp_list),
            desc="Processing HDFs using ray parallelization",
        ):
            processed_list.append(result)

        return processed_list

    def _merge_results(self, processed_list: list[pd.DataFrame]) -> pd.DataFrame:
        merged_result = pd.concat(processed_list).sort_values(["resample_rule", "time"])
        return merged_result

    def _filter_out_nan(self, merged_result: pd.DataFrame) -> pd.DataFrame:
        return merged_result.dropna().reset_index()

    def run(self) -> None:
        processed_list = self._process_using_ray()
        result = self._merge_results(processed_list)
        result = self._filter_out_nan(result)
        result.to_parquet(self._save_fp)
