from pathlib import Path
from typing import Literal

import pandas as pd
import ray
from ray.util import ActorPool
from tqdm import tqdm

from preprocess.after_merge_preprocessing_module import (
    AfterMergePreprocessingModuleBase,
)
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

        def process(self, hdf_fp: Path) -> pd.DataFrame:
            raw_df = self._read_hdf_with_path(hdf_fp)

            raw_preprocessed_df = self._run_raw_preprocessing_pipeline(raw_df)
            resample_preprocessed_df = self._run_resample_preprocessing_pipeline(raw_preprocessed_df)

            return resample_preprocessed_df

        def _run_raw_preprocessing_pipeline(self, raw_df: pd.DataFrame) -> pd.DataFrame:
            raw_preprocessed_df = raw_df
            for preprocessing_module in self._raw_preprocessing_pipeline:
                raw_preprocessed_df = preprocessing_module(raw_preprocessed_df)
            return raw_preprocessed_df

        def _run_resample_preprocessing_pipeline(self, raw_preprocessed_df: pd.DataFrame) -> pd.DataFrame:
            result_dfs = []
            for resample_rule in self._resample_rules:
                resampled_df = raw_preprocessed_df.resample(resample_rule)
                result_df = self._create_empty_resampled_df(raw_preprocessed_df, resample_rule)

                for preprocessing_module in self._resample_preprocessing_pipeline:
                    result_df = preprocessing_module(result_df, resampled_df)
                result_df["resample_rule"] = resample_rule
                result_dfs.append(result_df)

            resample_preprocessed_df = pd.concat(result_dfs)
            return resample_preprocessed_df

        def _create_empty_resampled_df(self, raw_preprocessed_df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
            resampled_index = (
                raw_preprocessed_df[~raw_preprocessed_df.index.duplicated(keep="first")]
                .resample(resample_rule)
                .asfreq()
                .index
            )
            empty_resampled_df = pd.DataFrame(index=resampled_index)
            return empty_resampled_df

    def __init__(
        self,
        raw_preprocessing_pipeline: list[RawPreprocessingModuleBase],
        resample_rules: list[ResampleRule],
        resample_preprocessing_pipeline: list[ResamplePreprocessingModuleBase],
        after_merge_preprocessing_pipeline: list[AfterMergePreprocessingModuleBase],
        raw_data_dir: str,
        ray_actor_num: int,
        save_fp: str,
    ) -> None:
        super().__init__()
        self._raw_data_fp_list = sorted(list(Path(raw_data_dir).glob("./*.h5")))
        self._save_fp = Path(save_fp)
        self._resample_rules = resample_rules
        self._ray_actor_num = ray_actor_num
        self._raw_preprocessing_pipeline = raw_preprocessing_pipeline
        self._resample_preprocessing_pipeline = resample_preprocessing_pipeline
        self._after_merge_preprocessing_pipeline = after_merge_preprocessing_pipeline

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

    def _run_after_merge_preprocessing_pipeline(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        after_merge_preprocessed_df = merged_df
        for preprocessing_module in self._after_merge_preprocessing_pipeline:
            after_merge_preprocessed_df = preprocessing_module(after_merge_preprocessed_df)
        return after_merge_preprocessed_df

    def run(self) -> None:
        processed_list = self._process_using_ray()
        result = self._merge_results(processed_list)
        result = self._run_after_merge_preprocessing_pipeline(result)
        result = result.sort_values(["resample_rule", "time"])
        result.to_parquet(self._save_fp)
