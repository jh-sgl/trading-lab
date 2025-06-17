from task.EXP0013_FactorFactory.preprocess.resample_preprocessing_module import (
    DateExtractor,
    OHLCVExtractor,
)
from task.EXP0013_GenFactor.preprocess.preprocessor import BasicPreprocessor
from task.EXP0013_GenFactor.preprocess.raw_preprocessing_module import (
    CumulativeTradeExtractor,
    DiffExtractor,
    TimeConverter,
    VolumeExtractor,
)

if __name__ == "__main__":
    raw_preprocessing_pipeline = [
        TimeConverter("time", "date", set_index_and_sort=True),
        VolumeExtractor(
            "future_long", "future_short", "future_volume", "future_cumulative_volume"
        ),
        CumulativeTradeExtractor("future_long", "future_short", "trade_cumsum"),
        *[
            DiffExtractor(
                f"m0s_top_tx_201_{i:+d}_cum_tx_vol", f"m0s_top_tx_201_{i:+d}_tx_vol"
            )
            for i in range(-20, 21)
        ],
        *[
            DiffExtractor(
                f"m0s_top_tx_201_{i:+d}_cum_tx_amt", f"m0s_top_tx_201_{i:+d}_tx_amt"
            )
            for i in range(-20, 21)
        ],
        *[
            DiffExtractor(
                f"m0s_top_tx_301_{i:+d}_cum_tx_vol", f"m0s_top_tx_301_{i:+d}_tx_vol"
            )
            for i in range(-20, 21)
        ],
        *[
            DiffExtractor(
                f"m0s_top_tx_301_{i:+d}_cum_tx_amt", f"m0s_top_tx_301_{i:+d}_tx_amt"
            )
            for i in range(-20, 21)
        ],
    ]

    resample_preprocessing_pipeline = [
        DateExtractor("date"),
        OHLCVExtractor("time", "time_boundary", select_ohlc="oc"),
        OHLCVExtractor("future_volume", "price", select_ohlc="v"),
        OHLCVExtractor("future_price", "price", select_ohlc="ohlc"),
        *[
            OHLCVExtractor(key, key, select_ohlc="ohlc")
            for key in [
                "future_basis",
                "future_theory",
                "openinterest",
                "individual_trade",
                "foreign_trade",
                "institutional_trade",
                "vkospi200_real",
                "165_price",
                "167_price",
                "usd_price",
                "trade_cumsum",
                "call_price",
                "put_price",
                "call_theory",
                "put_theory",
                "call_iv",
                "put_iv",
                "call_openint",
                "put_openint",
                "buy_order_total",
                "sell_order_total",
                "buy_order_1",
                "sell_order_1",
                "buy_order_2",
                "sell_order_2",
                "buy_order_3",
                "sell_order_3",
                "buy_order_4",
                "sell_order_4",
                "buy_order_5",
                "sell_order_5",
                "buy_order_1_price",
                "sell_order_1_price",
                "buy_order_2_price",
                "sell_order_2_price",
                "buy_order_3_price",
                "sell_order_3_price",
                "buy_order_4_price",
                "sell_order_4_price",
                "buy_order_5_price",
                "sell_order_5_price",
                "strike_price",
                "strike_call",
                "strike_put",
                "call_price_2nd_up",
                "call_theory_2nd_up",
                "call_iv_2nd_up",
                "call_openint_2nd_up",
                "put_price_2nd_down",
                "put_theory_2nd_down",
                "put_iv_2nd_down",
                "put_openint_2nd_down",
                "m0s_top_tx_201_strike_price",
                "m0s_top_tx_301_strike_price",
                *[f"m0s_top_tx_201_{i:+d}_price" for i in range(-20, 21)],
                *[f"m0s_top_tx_301_{i:+d}_price" for i in range(-20, 21)],
                *[f"m0s_top_tx_201_{i:+d}_tx_vol" for i in range(-20, 21)],
                *[f"m0s_top_tx_301_{i:+d}_tx_vol" for i in range(-20, 21)],
                *[f"m0s_top_tx_201_{i:+d}_tx_amt" for i in range(-20, 21)],
                *[f"m0s_top_tx_301_{i:+d}_tx_amt" for i in range(-20, 21)],
            ]
        ],
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
        raw_data_dir="/data/jh/repo/trading-lab/data/raw/20240624_fix_investors_to_cumsum_v2.2.0/h5",
        ray_actor_num=24,
        save_fp="/data/jh/repo/trading-lab/data/preprocessed_jh/v6_factor_factory.parquet",
    )

    basic_preprocessor.run()
