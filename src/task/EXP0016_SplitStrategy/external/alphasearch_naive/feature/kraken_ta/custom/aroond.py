# -*- coding: utf-8 -*-
from pandas import DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta import Imports
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_offset,
    verify_series,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    recent_maximum_index,
    recent_minimum_index,
)


def aroond(high, low, length=None, scalar=None, talib=None, offset=None, **kwargs):
    """Indicator: Aroon & Aroon Oscillator"""
    # Validate Arguments
    length = length if length and length > 0 else 14
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None:
        return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import AROON, AROONOSC

        aroon_down, aroon_up = AROON(high, low, length)
    else:
        raise RuntimeError("talib is not imported")

    # Handle fills
    if "fillna" in kwargs:
        aroon_up.fillna(kwargs["fillna"], inplace=True)
        aroon_down.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        aroon_up.fillna(method=kwargs["fill_method"], inplace=True)
        aroon_down.fillna(method=kwargs["fill_method"], inplace=True)

    # Offset
    if offset != 0:
        aroon_up = aroon_up.shift(offset)
        aroon_down = aroon_down.shift(offset)

    # Name and Categorize it
    aroon_down.name = f"AROOND_{length}"
    return aroon_down
