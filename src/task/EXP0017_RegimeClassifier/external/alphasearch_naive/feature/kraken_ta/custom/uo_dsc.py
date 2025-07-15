# -*- coding: utf-8 -*-
from pandas import DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta import Imports
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_drift,
    get_offset,
    verify_series,
)


def uo_dsc(
    high,
    low,
    close,
    fast=None,
    medium=None,
    slow=None,
    fast_w=None,
    medium_w=None,
    slow_w=None,
    talib=None,
    drift=None,
    offset=None,
    **kwargs,
):
    """Indicator: Ultimate Oscillator (UO)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 7
    fast_w = float(fast_w) if fast_w and fast_w > 0 else 4.0
    medium = int(medium) if medium and medium > 0 else 14
    medium_w = float(medium_w) if medium_w and medium_w > 0 else 2.0
    slow = int(slow) if slow and slow > 0 else 28
    slow_w = float(slow_w) if slow_w and slow_w > 0 else 1.0
    if not medium < fast:
        raise RuntimeError()
    if not slow < medium:
        raise RuntimeError()
    _length = max(fast, medium, slow)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None:
        return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ULTOSC

        uo = ULTOSC(high, low, close, fast, medium, slow)
    else:
        raise RuntimeError("talibs are not imported")
    # Offset
    if offset != 0:
        uo = uo.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        uo.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        uo.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    uo.name = f"UODSC_{fast}_{medium}_{slow}"
    uo.category = "momentum"

    return uo
