# -*- coding: utf-8 -*-
from pandas import DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.trend.decreasing import (
    decreasing,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.trend.increasing import (
    increasing,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    ma,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_offset,
    verify_series,
)


def amatlr(close=None, fast=None, slow=None, mamode=None, offset=None, **kwargs):
    """Indicator: Archer Moving Averages Trends (AMAT)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 8
    slow = int(slow) if slow and slow > 0 else 21
    if not slow > fast:
        raise RuntimeError()
    lookback = fast
    mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    close = verify_series(close, max(fast, slow, lookback))
    offset = get_offset(offset)

    if "length" in kwargs:
        kwargs.pop("length")

    if close is None:
        return

    # # Calculate Result
    fast_ma = ma(mamode, close, length=fast, **kwargs)
    slow_ma = ma(mamode, close, length=slow, **kwargs)

    mas_long = increasing(fast_ma, lookback) & decreasing(slow_ma, lookback)

    # Offset
    if offset != 0:
        mas_long = mas_long.shift(offset)

    # # Handle fills
    if "fillna" in kwargs:
        mas_long.fillna(kwargs["fillna"], inplace=True)

    if "fill_method" in kwargs:
        mas_long.fillna(method=kwargs["fill_method"], inplace=True)

    mas_long.name = f"AMAT{mamode[0]}_LR_{fast}_{slow}_{lookback}"
    mas_long.category = "trend"

    return mas_long
