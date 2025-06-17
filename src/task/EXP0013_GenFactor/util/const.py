import pandas as pd


class DFKey:
    # Original columns
    DATE: tuple[str, str] = ("date", "__NA__")
    RESAMPLE_RULE: tuple[str, str] = ("resample_rule", "__NA__")

    FUTURE_PRICE_OPEN: tuple[str, str] = ("price", "open")
    FUTURE_PRICE_CLOSE: tuple[str, str] = ("price", "close")
    TRADE_CUMSUM_CLOSE: tuple[str, str] = ("trade_cumsum", "close")

    FUTURE_THEORY_CLOSE: tuple[str, str] = ("future_theory", "close")
    FUTURE_BASIS_CLOSE: tuple[str, str] = ("future_basis", "close")

    FOREIGN_TRADE_CLOSE: tuple[str, str] = ("foreign_trade", "close")
    INDIVIDUAL_TRADE_CLOSE: tuple[str, str] = ("individual_trade", "close")
    INSTITUTIONAL_TRADE_CLOSE: tuple[str, str] = ("institutional_trade", "close")

    CALL_IV_CLOSE: tuple[str, str] = ("call_iv", "close")
    CALL_IV_2ND_UP_CLOSE: tuple[str, str] = ("call_iv_2nd_up", "close")
    CALL_OPENINT_CLOSE: tuple[str, str] = ("call_openint", "close")
    CALL_OPENINT_2ND_UP_CLOSE: tuple[str, str] = ("call_openint_2nd_up", "close")

    PUT_IV_CLOSE: tuple[str, str] = ("put_iv", "close")
    PUT_IV_2ND_DOWN_CLOSE: tuple[str, str] = ("put_iv_2nd_down", "close")
    PUT_OPENINT_CLOSE: tuple[str, str] = ("put_openint", "close")
    PUT_OPENINT_2ND_DOWN_CLOSE: tuple[str, str] = ("put_openint_2nd_down", "close")

    VKOSPI200_REAL_CLOSE: tuple[str, str] = ("vkospi200_real", "close")

    BOND_167_CLOSE: tuple[str, str] = ("167_price", "close")
    BOND_165_CLOSE: tuple[str, str] = ("165_price", "close")
    USD_PRICE_CLOSE: tuple[str, str] = ("usd_price", "close")

    BUY_ORDER_1_CLOSE: tuple[str, str] = ("buy_order_1", "close")
    BUY_ORDER_2_CLOSE: tuple[str, str] = ("buy_order_2", "close")
    BUY_ORDER_3_CLOSE: tuple[str, str] = ("buy_order_3", "close")
    BUY_ORDER_4_CLOSE: tuple[str, str] = ("buy_order_4", "close")
    BUY_ORDER_5_CLOSE: tuple[str, str] = ("buy_order_5", "close")

    SELL_ORDER_1_CLOSE: tuple[str, str] = ("sell_order_1", "close")
    SELL_ORDER_2_CLOSE: tuple[str, str] = ("sell_order_2", "close")
    SELL_ORDER_3_CLOSE: tuple[str, str] = ("sell_order_3", "close")
    SELL_ORDER_4_CLOSE: tuple[str, str] = ("sell_order_4", "close")
    SELL_ORDER_5_CLOSE: tuple[str, str] = ("sell_order_5", "close")

    BUY_ORDER_1_PRICE: tuple[str, str] = ("buy_order_1_price", "close")
    BUY_ORDER_2_PRICE: tuple[str, str] = ("buy_order_2_price", "close")
    BUY_ORDER_3_PRICE: tuple[str, str] = ("buy_order_3_price", "close")
    BUY_ORDER_4_PRICE: tuple[str, str] = ("buy_order_4_price", "close")
    BUY_ORDER_5_PRICE: tuple[str, str] = ("buy_order_5_price", "close")

    SELL_ORDER_1_PRICE: tuple[str, str] = ("sell_order_1_price", "close")
    SELL_ORDER_2_PRICE: tuple[str, str] = ("sell_order_2_price", "close")
    SELL_ORDER_3_PRICE: tuple[str, str] = ("sell_order_3_price", "close")
    SELL_ORDER_4_PRICE: tuple[str, str] = ("sell_order_4_price", "close")
    SELL_ORDER_5_PRICE: tuple[str, str] = ("sell_order_5_price", "close")

    # Newly added columns for labeling
    PROFIT_AT_MARKET_CLOSE: tuple[str, str] = ("label", "market_close")

    # Newly added columns for backtesting and statistics
    OUTPUT_PRED_SHORT: tuple[str, str] = ("output_pred", "short")
    OUTPUT_PRED_HOLD: tuple[str, str] = ("output_pred", "hold")
    OUTPUT_PRED_LONG: tuple[str, str] = ("output_pred", "long")
    SIGNAL: tuple[str, str] = ("backtest", "signal")
    PRICE_EXECUTION: tuple[str, str] = ("backtest", "price_execution")
    DAILY_LAST_ROW: tuple[str, str] = ("backtest", "daily_last_row")
    ENTRY_TIME: tuple[str, str] = ("backtest", "entry_time")
    EXIT_TIME: tuple[str, str] = ("backtest", "exit_time")
    ENTRY_PRICE: tuple[str, str] = ("backtest", "entry_price")
    EXIT_PRICE: tuple[str, str] = ("backtest", "exit_price")
    ENTRY_SIGNAL: tuple[str, str] = ("backtest", "entry_signal")
    RAW_PNL: tuple[str, str] = ("backtest", "raw_pnl")
    NET_PNL: tuple[str, str] = ("backtest", "net_pnl")
    RAW_PNL_UWON: tuple[str, str] = ("backtest", "raw_pnl_uwon")
    NET_PNL_UWON: tuple[str, str] = ("backtest", "net_pnl_uwon")
    COST: tuple[str, str] = ("backtest", "cost")
    MAX_POSITION_SIZE: tuple[str, str] = ("backtest", "max_position_size")


class Num:
    COMMISSION_RATE = 0.000029
    SLIPPAGE_PER_EXECUTION = 0.05
    EPS = 1e-25
    UWON = 100_000_000
    PRICE_MULTIPLIER = 250_000


def _load_margin_rate_config() -> pd.DataFrame:
    MARGIN_RATE_PATH = "/data/jh/Live4Common/csv/margin_rate.csv"
    margin_rate = pd.read_csv(MARGIN_RATE_PATH)
    margin_rate = margin_rate.set_index("date")
    margin_rate.index = pd.to_datetime(margin_rate.index)
    margin_rate = margin_rate.sort_index()
    return margin_rate


MARGIN_RATE = _load_margin_rate_config()
