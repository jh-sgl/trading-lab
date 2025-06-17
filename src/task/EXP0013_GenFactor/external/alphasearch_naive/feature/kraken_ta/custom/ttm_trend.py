# -*- coding: utf-8 -*-
from pandas import DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    hl2,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_offset,
    verify_series,
)


def ttm_trend(high, low, close, length=None, offset=None, **kwargs):
    """Indicator: TTM Trend (TTM_TRND)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 6
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return

    # Calculate Result
    trend_avg = hl2(high, low)
    for i in range(1, length):
        trend_avg = trend_avg + hl2(high.shift(i), low.shift(i))

    trend_avg = trend_avg / length

    tm_trend = (close > trend_avg).astype(int)
    tm_trend.replace(0, -1, inplace=True)

    # Offset
    if offset != 0:
        tm_trend = tm_trend.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        tm_trend.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        tm_trend.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    tm_trend.name = f"TTM_TRND_{length}"
    tm_trend.category = "momentum"

    # Prepare DataFrame to return
    data = {tm_trend.name: tm_trend}
    df = DataFrame(data)
    df.name = f"TTMTREND_{length}"
    df.category = tm_trend.category

    return df
