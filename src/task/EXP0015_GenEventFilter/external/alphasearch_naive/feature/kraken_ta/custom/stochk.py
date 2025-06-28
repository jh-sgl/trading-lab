# -*- coding: utf-8 -*-
from pandas import DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    ma,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_offset,
    non_zero_range,
    verify_series,
)


def stochk(high, low, close, k=None, smooth_k=None, mamode=None, offset=None, **kwargs):
    """Indicator: Stochastic Oscillator (STOCH)"""
    # Validate arguments
    k = k if k and k > 0 else 14
    smooth_k = smooth_k if smooth_k and smooth_k > 0 else 3
    _length = max(k, smooth_k)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "sma"

    if high is None or low is None or close is None:
        return

    # Calculate Result
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()

    stoch = 100 * (close - lowest_low)
    stoch /= non_zero_range(highest_high, lowest_low)

    stoch_k = ma(mamode, stoch.loc[stoch.first_valid_index() :,], length=smooth_k)

    # Offset
    if offset != 0:
        stoch_k = stoch_k.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        stoch_k.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stoch_k.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _name = "STOCH"
    _props = f"_{k}_{smooth_k}"
    stoch_k.name = f"{_name}k{_props}"
    stoch_k.category = "momentum"

    # Prepare DataFrame to return
    data = {stoch_k.name: stoch_k}
    df = DataFrame(data)
    df.name = f"{_name}k{_props}"
    df.category = stoch_k.category
    return df
