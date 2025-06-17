# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta import Imports
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    ema,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_offset,
    verify_series,
    signals,
)


def macd(close, fast=None, slow=None, signal=None, talib=None, offset=None, **kwargs):
    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    if not slow > fast:
        raise RuntimeError()
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return

    as_mode = kwargs.setdefault("asmode", False)

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MACD

        macd, signalma, histogram = MACD(close, fast, slow, signal)
    else:
        fastma = ema(close, length=fast)
        slowma = ema(close, length=slow)

        macd = fastma - slowma
        signalma = ema(close=macd.loc[macd.first_valid_index() :,], length=signal)
        histogram = macd - signalma

    if as_mode:
        macd = macd - signalma
        signalma = ema(close=macd.loc[macd.first_valid_index() :,], length=signal)
        histogram = macd - signalma

    # Offset
    if offset != 0:
        macd = macd.shift(offset)
        histogram = histogram.shift(offset)
        signalma = signalma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        macd.fillna(kwargs["fillna"], inplace=True)
        histogram.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        macd.fillna(method=kwargs["fill_method"], inplace=True)
        histogram.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _asmode = "AS" if as_mode else ""
    _props = f"_{fast}_{slow}_{signal}"
    macd.name = f"MACD{_asmode}{_props}"
    histogram.name = f"MACD{_asmode}h{_props}"
    signalma.name = f"MACD{_asmode}s{_props}"
    macd.category = histogram.category = signalma.category = "momentum"
    return histogram
