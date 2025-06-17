# -*- coding: utf-8 -*-
from pandas import DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    ema,
    ma,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_drift,
    get_offset,
    verify_series,
)


def tsisig(
    close,
    fast=None,
    slow=None,
    signal=None,
    scalar=None,
    mamode=None,
    drift=None,
    offset=None,
    **kwargs,
):
    """Indicator: True Strength Index (TSI)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 13
    slow = int(slow) if slow and slow > 0 else 25
    signal = int(signal) if signal and signal > 0 else 13
    if not slow > fast:
        raise RuntimeError()
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, max(fast, slow))
    drift = get_drift(drift)
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "ema"
    if "length" in kwargs:
        kwargs.pop("length")

    if close is None:
        return

    # Calculate Result
    diff = close.diff(drift)
    slow_ema = ema(close=diff, length=slow, **kwargs)
    fast_slow_ema = ema(close=slow_ema, length=fast, **kwargs)

    abs_diff = diff.abs()
    abs_slow_ema = ema(close=abs_diff, length=slow, **kwargs)
    abs_fast_slow_ema = ema(close=abs_slow_ema, length=fast, **kwargs)

    tsi = scalar * fast_slow_ema / abs_fast_slow_ema
    tsi_signal = ma(mamode, tsi, length=signal)

    # Offset
    if offset != 0:
        tsi = tsi.shift(offset)
        tsi_signal = tsi_signal.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        tsi.fillna(kwargs["fillna"], inplace=True)
        tsi_signal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        tsi.fillna(method=kwargs["fill_method"], inplace=True)
        tsi_signal.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    tsi_signal.name = f"TSIs_{fast}_{slow}_{signal}"
    tsi_signal.category = "momentum"

    return tsi_signal
