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


def aroonosc(high, low, length=None, scalar=None, talib=None, offset=None, **kwargs):
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
        from talib import AROONOSC

        aroon_osc = AROONOSC(high, low, length)
    else:
        raise RuntimeError("talib is not imported")

    # Handle fills
    if "fillna" in kwargs:
        aroon_osc.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        aroon_osc.fillna(method=kwargs["fill_method"], inplace=True)

    # Offset
    if offset != 0:
        aroon_osc = aroon_osc.shift(offset)

    # Name and Categorize it
    aroon_osc.name = f"AROONOSC_{length}"
    return aroon_osc
