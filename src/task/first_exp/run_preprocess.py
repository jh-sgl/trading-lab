from preprocess.after_merge_preprocessing_module import DailyShifter
from preprocess.preprocessor import BasicPreprocessor
from preprocess.raw_preprocessing_module import TimeConverter, VolumeExtractor
from preprocess.resample_preprocessing_module import (
    DateExtractor,
    DeltaExtractor,
    MarketClosingPriceAssigner,
    MaxCrossingBandExtractor,
    OHLCVExtractor,
)

# TODO: use hydra
# TODO: convert implicit suffix / prefix to explicit arguments
if __name__ == "__main__":
    raw_preprocessing_pipeline = [
        TimeConverter("time", "date", set_index_and_sort=True),
        VolumeExtractor("future_long", "future_short", "future_cumulative_volume", "future_volume"),
    ]

    resample_preprocessing_pipeline = (
        [
            DateExtractor("date"),
            DeltaExtractor("future_price", "future_price"),
            OHLCVExtractor("future_volume", "future_volume", select_ohlc="v"),
        ]
        + [
            OHLCVExtractor(key, key, select_ohlc="ohlc")
            for key in [
                "future_price",
                "openinterest",
                "foreign_trade",
                "institutional_trade",
                "vkospi200_real",
                "167_price",
                "usd_price",
            ]
        ]
        + [
            MarketClosingPriceAssigner("future_price_close", "future_price_market_closing"),
            MaxCrossingBandExtractor("future_price", "future_price", "date", band_width=0.2),
        ]
    )

    after_merge_preprocessing_pipeline = [DailyShifter("tr", "prev_tr", "date", pd_shift_args={})]

    basic_preprocessor = BasicPreprocessor(
        raw_preprocessing_pipeline=raw_preprocessing_pipeline,
        resample_preprocessing_pipeline=resample_preprocessing_pipeline,
        after_merge_preprocessing_pipeline=after_merge_preprocessing_pipeline,
        resample_rules=[
            "1min",
            "5min",
            "15min",
            "1h",
        ],
        raw_data_dir="/data/jh/repo/trading-lab/data/raw/20240624_fix_investors_to_cumsum_v2.2.0/h5",
        ray_actor_num=96,
        save_fp="/data/jh/repo/trading-lab/data/preprocessed_jh/v3.parquet",
    )

    basic_preprocessor.run()
