import copy
import random
import inspect
import numpy as np
import pandas as pd
from collections import *

import task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.custom as kta
import task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta as pta
from task.EXP0013_GenFactor.external.alphasearch_naive.feature.kraken_ta.config import (
    kta_ohlc,
    kta_volume,
)


def eval_res(X, function, map_name):
    try:
        res = eval(function)
    except Exception as e:
        print(f"Error:  Function: {function}")
        raise Exception(e)
    res = pd.DataFrame(res, index=X.index)

    ta_cols = list(res.columns)
    if len(ta_cols) > 1:
        print(function)
        print(f"Too many TAs are returned: {len(ta_cols)}")
        res = pd.DataFrame(res.iloc[:, 0])

    res = res.rename(columns={res.columns[0]: map_name + "_" + res.columns[0]})
    return res


def run_fn(data, fn, map_name="price"):
    X = data[map_name].copy()
    res = eval_res(X, fn, map_name)

    if isinstance(res, pd.DataFrame):
        ta_cols = list(res.columns)

    X = pd.concat([X, res], axis=1)
    return X, ta_cols


def add_ta(data_, ta_info):
    ta_cols = []
    if not isinstance(data_, pd.DataFrame):
        data = data_["df"]
    else:
        data = data_

    if "map_name" in list(ta_info.keys()):
        for fn, map_name in zip(
            list(ta_info["fn"].values()), list(ta_info["map_name"].values())
        ):
            output, ta_col = run_fn(data, fn, map_name=map_name)
            data["ta", ta_col[0]] = output[ta_col[0]]
            ta_cols.append(["ta", ta_col[0]])
    else:
        for fn in list(ta_info["fn"].values()):
            output, ta_col = run_fn(data, fn, map_name="price")
            data["ta", ta_col[0]] = output[ta_col[0]]
            ta_cols.append(["ta", ta_col[0]])
    return data, ta_cols
