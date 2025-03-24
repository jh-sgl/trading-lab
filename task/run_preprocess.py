from jh.preprocess.preprocessor import BasicPreprocessor
from jh.preprocess.raw_preprocessing_module import TimeConverter, VolumeExtractor
from jh.preprocess.resample_preprocessing_module import DeltaExtractor, OHLCVExtractor

if __name__ == "__main__":
    raw_preprocessing_pipeline = [
        TimeConverter("time", set_index_and_sort=True),
        VolumeExtractor(
            "future_long", "future_short", "future_cumulative_volume", "future_volume"
        ),
    ]
    resample_preprocessing_pipeline = [
        DeltaExtractor("future_price", "future_price"),
        OHLCVExtractor("future_volume", "future_volume", select_ohlc="v"),
    ] + [
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
    basic_preprocessor = BasicPreprocessor(
        raw_preprocessing_pipeline=raw_preprocessing_pipeline,
        resample_preprocessing_pipeline=resample_preprocessing_pipeline,
        resample_rules=[
            "1min",
            "5min",
            "15min",
            "1h",
        ],
        raw_data_dir="/data/jh/AlphaSearch_Naive/data/raw/20240624_fix_investors_to_cumsum_v2.2.0/h5",
        ray_actor_num=64,
        save_fp="/data/jh/AlphaSearch_Naive/data/preprocessed_jh/v2.parquet",
    )

    basic_preprocessor.run()
