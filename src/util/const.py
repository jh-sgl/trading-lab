import pandas as pd


class Num:
    COMMISSION_RATE = 0.000029
    SLIPPAGE_PER_EXECUTION = 0.05
    EPS = 1e-25
    UWON = 100_000_000
    PRICE_MULTIPLIER = 250_000


def _load_margin_rate_config() -> pd.DataFrame:
    MARGIN_RATE_PATH = "/data/jh/Live4Common/csv/margin_rate.csv"
    margin_rate = pd.read_csv(MARGIN_RATE_PATH)
    margin_rate = margin_rate.set_index("date")
    margin_rate.index = pd.to_datetime(margin_rate.index)
    margin_rate = margin_rate.sort_index()
    return margin_rate


MARGIN_RATE = _load_margin_rate_config()
