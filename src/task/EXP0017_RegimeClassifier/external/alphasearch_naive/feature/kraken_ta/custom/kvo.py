# -*- coding: utf-8 -*-
from pandas import DataFrame
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.overlap import (
    hlc3,
    ma,
)
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_drift,
    get_offset,
    signed_series,
    verify_series,
)


def kvo(
    high,
    low,
    close,
    volume,
    fast=None,
    slow=None,
    mamode=None,
    drift=None,
    offset=None,
    **kwargs,
):
    """Indicator: Klinger Volume Oscillator (KVO)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 34
    slow = int(slow) if slow and slow > 0 else 55
    if not slow > fast:
        raise RuntimeError()
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "ema"
    _length = max(fast, slow)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None or volume is None:
        return

    # Calculate Result
    signed_volume = volume * signed_series(hlc3(high, low, close), 1)
    sv = signed_volume.loc[signed_volume.first_valid_index() :,]
    kvo = ma(mamode, sv, length=fast) - ma(mamode, sv, length=slow)

    # Offset
    if offset != 0:
        kvo = kvo.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        kvo.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        kvo.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _props = f"_{fast}_{slow}"
    kvo.name = f"KVO{_props}"
    kvo.category = "volume"
    return kvo
