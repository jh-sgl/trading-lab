# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta import Imports
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    rma,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_drift,
    get_offset,
    verify_series,
    signals,
)


def rsi(close, length=None, scalar=None, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: Relative Strength Index (RSI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import RSI

        rsi = RSI(close, length)
    else:
        raise RuntimeError("talib is not imported")

    # Offset
    if offset != 0:
        rsi = rsi.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        rsi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rsi.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    rsi.name = f"RSI_{length}"
    rsi.category = "momentum"
    return rsi
