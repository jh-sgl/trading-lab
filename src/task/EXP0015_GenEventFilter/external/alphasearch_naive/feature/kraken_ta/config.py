# Series data
tune_series = ["open", "high", "low", "close", "volume"]

# Parameters to tune
tune_params = [
    "acceleration",
    "accelerationlong",
    "accelerationshort",
    "atr_length",
    "atr_period",
    "average_lenght",
    "average_length",
    "bb_length",
    "channel_lenght",
    "channel_length",
    "chikou_period",
    "d",
    "ddof",
    "ema_fast",
    "ema_slow",
    "er",
    "fast",
    "fast_period",
    "fastd_period",
    "fastk_period",
    "fastperiod",
    "high_length",
    "jaw",
    "k",
    "k_period",
    "kc_length",
    "kijun",
    "kijun_period",
    "length",
    "lensig",
    "lips",
    "long",
    "long_period",
    "lookback",
    "low_length",
    "lower_length",
    "lower_period",
    "ma_length",
    # "mamode",
    "max_lookback",
    "maxperiod",
    "medium",
    "min_lookback",
    "minperiod",
    "mom_length",
    "mom_smooth",
    "p",
    "period",
    "period_fast",
    "period_slow",
    "q",
    "r1",
    "r2",
    "r3",
    "r4",
    "roc1",
    "roc2",
    "roc3",
    "roc4",
    "rsi_length",
    "rsi_period",
    "run_length",
    "rvi_length",
    "senkou",
    "senkou_period",
    "short",
    "short_period",
    "signal",
    "signalperiod",
    "slow",
    "slow_period",
    "slowd_period",
    "slowk_period",
    "slowperiod",
    "sma1",
    "sma2",
    "sma3",
    "sma4",
    "smooth",
    "smooth_k",
    "stoch_period",
    "swma_length",
    "tclength",
    "teeth",
    "tenkan",
    "tenkan_period",
    "timeperiod",
    "timeperiod1",
    "timeperiod2",
    "timeperiod3",
    "upper_length",
    "upper_period",
    "width",
    "wma_period",
]

pandas_ta_mamodes = {
    "dema": 0,
    "ema": 1,
    "fwma": 2,
    # "hma": 3,  # Issue with low range https://github.com/jmrichardson/tuneta/pull/24
    "linreg": 4,
    "midpoint": 5,
    "pwma": 6,
    "rma": 7,
    "sinwma": 8,
    "sma": 9,
    "swma": 10,
    "t3": 11,
    "tema": 12,
    "trima": 13,
    "vidya": 14,
    "wma": 15,
    "zlma": 16,
}

kta_ohlc_tiny = [
    "kta.fisher",
    "kta.rsi",
    "kta.uo_asc",
]

kta_ohlc_picked = [
    "kta.bias",
    "kta.uo_asc",
    "kta.uo_dsc",
    "kta.fisher",
    "kta.ao",
    "kta.aroonosc",
    "kta.aroond",
    "kta.aroonu",
    "kta.ttm_trend",
    "kta.tsiosc",
    "kta.rsi",
    "kta.macd",
]

kta_ohlc_xpicked = [
    "kta.amatlr",
    "kta.amatsr",
    "kta.cci",
    "kta.rvi",
    "kta.stochd",
    "kta.stochk",
    "kta.stochrsid",
    "kta.stochrsik",
    "kta.tsi",
    "kta.tsisig",
]

kta_close_only = [
    "kta.bias",
    "kta.fisher",
    "kta.rsi",
    "kta.macd",
    "kta.tsiosc",
    "kta.stochrsid",
    "kta.stochrsik",
]

kta_ohlc = [
    "kta.amatlr",
    "kta.amatsr",
    "kta.ao",
    "kta.aroond",
    "kta.aroonosc",
    "kta.aroonu",
    "kta.bias",
    "kta.cci",
    "kta.fisher",
    "kta.rsi",
    "kta.rvi",
    "kta.stochd",
    "kta.stochk",
    "kta.stochrsid",
    "kta.stochrsik",
    "kta.tsi",
    "kta.tsiosc",
    "kta.tsisig",
    "kta.ttm_trend",
    "kta.uo_asc",
    "kta.uo_dsc",
]

kta_volume = [
    "kta.adosc",
    "kta.efi",
    "kta.kvo",
    "kta.mfi",
    "kta.kvosig",
]
