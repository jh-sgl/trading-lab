# -*- coding: utf-8 -*-
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.utils import (
    get_drift,
    get_offset,
    is_percent,
    verify_series,
)


def decreasing(
    close,
    length=None,
    strict=None,
    asint=None,
    percent=None,
    drift=None,
    offset=None,
    **kwargs,
):
    """Indicator: Decreasing"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 1
    strict = strict if isinstance(strict, bool) else False
    asint = asint if isinstance(asint, bool) else True
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    percent = float(percent) if is_percent(percent) else False

    if close is None:
        return

    # Calculate Result
    close_ = (1 - 0.01 * percent) * close if percent else close
    if strict:
        # Returns value as float64? Have to cast to bool
        decreasing = close < close_.shift(drift)
        for x in range(3, length + 1):
            decreasing = decreasing & (
                close.shift(x - (drift + 1)) < close_.shift(x - drift)
            )

        decreasing.fillna(0, inplace=True)
        decreasing = decreasing.astype(bool)
    else:
        decreasing = close_.diff(length) < 0

    if asint:
        decreasing = decreasing.astype(int)

    # Offset
    if offset != 0:
        decreasing = decreasing.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        decreasing.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        decreasing.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _percent = f"_{0.01 * percent}" if percent else ""
    _props = f"{'S' if strict else ''}DEC{'p' if percent else ''}"
    decreasing.name = f"{_props}_{length}{_percent}"
    decreasing.category = "trend"

    return decreasing


decreasing.__doc__ = """Decreasing

Returns True if the series is decreasing over a period, False otherwise.
If the kwarg 'strict' is True, it returns True if it is continuously decreasing
over the period. When using the kwarg 'asint', then it returns 1 for True
or 0 for False.

Calculation:
    if strict:
        decreasing = all(i > j for i, j in zip(close[-length:], close[1:]))
    else:
        decreasing = close.diff(length) < 0

    if asint:
        decreasing = decreasing.astype(int)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    strict (bool): If True, checks if the series is continuously decreasing over the period. Default: False
    percent (float): Percent as an integer. Default: None
    asint (bool): Returns as binary. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
