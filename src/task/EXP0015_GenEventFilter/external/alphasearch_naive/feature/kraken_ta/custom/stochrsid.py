# -*- coding: utf-8 -*-
from pandas import DataFrame
from .rsi import rsi
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    ma,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_offset,
    non_zero_range,
    verify_series,
)


def stochrsid(
    close,
    rsi_length=None,
    k=None,
    d=None,
    mamode=None,
    offset=None,
    **kwargs,
):
    """Indicator: Stochastic RSI Oscillator (STOCHRSI)"""
    # Validate arguments
    length = rsi_length
    rsi_length = rsi_length if rsi_length and rsi_length > 0 else 14
    k = k if k and k > 0 else 3
    d = d if d and d > 0 else 3
    close = verify_series(close, max(length, rsi_length, k, d))
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "sma"

    if close is None:
        return

    # Calculate Result
    rsi_ = rsi(close, length=rsi_length)
    lowest_rsi = rsi_.rolling(length).min()
    highest_rsi = rsi_.rolling(length).max()

    stoch = 100 * (rsi_ - lowest_rsi)
    stoch /= non_zero_range(highest_rsi, lowest_rsi)

    stochrsi_k = ma(mamode, stoch, length=k)
    stochrsi_d = ma(mamode, stochrsi_k, length=d)

    # Offset
    if offset != 0:
        stochrsi_k = stochrsi_k.shift(offset)
        stochrsi_d = stochrsi_d.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        stochrsi_k.fillna(kwargs["fillna"], inplace=True)
        stochrsi_d.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stochrsi_k.fillna(method=kwargs["fill_method"], inplace=True)
        stochrsi_d.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _name = "STOCHRSI"
    _props = f"_{length}_{rsi_length}_{k}_{d}"
    stochrsi_d.name = f"{_name}d{_props}"
    stochrsi_d.category = "momentum"

    return stochrsi_d
