import pandas as pd


class DFKey:
    # Original columns
    DATE: tuple[str, str] = ("date", "__NA__")
    RESAMPLE_RULE: tuple[str, str] = ("resample_rule", "__NA__")

    # Newly added columns for labeling
    SHARPE_TO_X_MIN: tuple[str, str] = ("label", "sharpe_at_market_close")
    PROFIT_AT_MARKET_CLOSE: tuple[str, str] = ("label", "profit_at_market_close")

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

    # Data columns
    FUTURE_PRICE_OPEN: tuple[str, str] = ("price", "open")
    FUTURE_PRICE_HIGH: tuple[str, str] = ("price", "high")
    FUTURE_PRICE_LOW: tuple[str, str] = ("price", "low")
    FUTURE_PRICE_CLOSE: tuple[str, str] = ("price", "close")
    FUTURE_VOLUME: tuple[str, str] = ("price", "volume")

    TRADE_CUMSUM_OPEN: tuple[str, str] = ("trade_cumsum", "open")
    TRADE_CUMSUM_HIGH: tuple[str, str] = ("trade_cumsum", "high")
    TRADE_CUMSUM_LOW: tuple[str, str] = ("trade_cumsum", "low")
    TRADE_CUMSUM_CLOSE: tuple[str, str] = ("trade_cumsum", "close")

    FUTURE_THEORY_OPEN: tuple[str, str] = ("future_theory", "open")
    FUTURE_THEORY_HIGH: tuple[str, str] = ("future_theory", "high")
    FUTURE_THEORY_LOW: tuple[str, str] = ("future_theory", "low")
    FUTURE_THEORY_CLOSE: tuple[str, str] = ("future_theory", "close")

    FUTURE_BASIS_OPEN: tuple[str, str] = ("future_basis", "open")
    FUTURE_BASIS_HIGH: tuple[str, str] = ("future_basis", "high")
    FUTURE_BASIS_LOW: tuple[str, str] = ("future_basis", "low")
    FUTURE_BASIS_CLOSE: tuple[str, str] = ("future_basis", "close")

    FOREIGN_TRADE_OPEN: tuple[str, str] = ("foreign_trade", "open")
    FOREIGN_TRADE_HIGH: tuple[str, str] = ("foreign_trade", "high")
    FOREIGN_TRADE_LOW: tuple[str, str] = ("foreign_trade", "low")
    FOREIGN_TRADE_CLOSE: tuple[str, str] = ("foreign_trade", "close")

    INDIVIDUAL_TRADE_OPEN: tuple[str, str] = ("individual_trade", "open")
    INDIVIDUAL_TRADE_HIGH: tuple[str, str] = ("individual_trade", "high")
    INDIVIDUAL_TRADE_LOW: tuple[str, str] = ("individual_trade", "low")
    INDIVIDUAL_TRADE_CLOSE: tuple[str, str] = ("individual_trade", "close")

    INSTITUTIONAL_TRADE_OPEN: tuple[str, str] = ("institutional_trade", "open")
    INSTITUTIONAL_TRADE_HIGH: tuple[str, str] = ("institutional_trade", "high")
    INSTITUTIONAL_TRADE_LOW: tuple[str, str] = ("institutional_trade", "low")
    INSTITUTIONAL_TRADE_CLOSE: tuple[str, str] = ("institutional_trade", "close")

    CALL_PRICE_OPEN: tuple[str, str] = ("call_price", "open")
    CALL_PRICE_HIGH: tuple[str, str] = ("call_price", "high")
    CALL_PRICE_LOW: tuple[str, str] = ("call_price", "low")
    CALL_PRICE_CLOSE: tuple[str, str] = ("call_price", "close")

    PUT_PRICE_OPEN: tuple[str, str] = ("put_price", "open")
    PUT_PRICE_HIGH: tuple[str, str] = ("put_price", "high")
    PUT_PRICE_LOW: tuple[str, str] = ("put_price", "low")
    PUT_PRICE_CLOSE: tuple[str, str] = ("put_price", "close")

    CALL_IV_OPEN: tuple[str, str] = ("call_iv", "open")
    CALL_IV_HIGH: tuple[str, str] = ("call_iv", "high")
    CALL_IV_LOW: tuple[str, str] = ("call_iv", "low")
    CALL_IV_CLOSE: tuple[str, str] = ("call_iv", "close")

    PUT_IV_OPEN: tuple[str, str] = ("put_iv", "open")
    PUT_IV_HIGH: tuple[str, str] = ("put_iv", "high")
    PUT_IV_LOW: tuple[str, str] = ("put_iv", "low")
    PUT_IV_CLOSE: tuple[str, str] = ("put_iv", "close")

    CALL_OPENINT_OPEN: tuple[str, str] = ("call_openint", "open")
    CALL_OPENINT_HIGH: tuple[str, str] = ("call_openint", "high")
    CALL_OPENINT_LOW: tuple[str, str] = ("call_openint", "low")
    CALL_OPENINT_CLOSE: tuple[str, str] = ("call_openint", "close")

    PUT_OPENINT_OPEN: tuple[str, str] = ("put_openint", "open")
    PUT_OPENINT_HIGH: tuple[str, str] = ("put_openint", "high")
    PUT_OPENINT_LOW: tuple[str, str] = ("put_openint", "low")
    PUT_OPENINT_CLOSE: tuple[str, str] = ("put_openint", "close")

    CALL_PRICE_2ND_UP_OPEN: tuple[str, str] = ("call_price_2nd_up", "open")
    CALL_PRICE_2ND_UP_HIGH: tuple[str, str] = ("call_price_2nd_up", "high")
    CALL_PRICE_2ND_UP_LOW: tuple[str, str] = ("call_price_2nd_up", "low")
    CALL_PRICE_2ND_UP_CLOSE: tuple[str, str] = ("call_price_2nd_up", "close")

    PUT_PRICE_2ND_DOWN_OPEN: tuple[str, str] = ("put_price_2nd_down", "open")
    PUT_PRICE_2ND_DOWN_HIGH: tuple[str, str] = ("put_price_2nd_down", "high")
    PUT_PRICE_2ND_DOWN_LOW: tuple[str, str] = ("put_price_2nd_down", "low")
    PUT_PRICE_2ND_DOWN_CLOSE: tuple[str, str] = ("put_price_2nd_down", "close")

    CALL_IV_2ND_UP_OPEN: tuple[str, str] = ("call_iv_2nd_up", "open")
    CALL_IV_2ND_UP_HIGH: tuple[str, str] = ("call_iv_2nd_up", "high")
    CALL_IV_2ND_UP_LOW: tuple[str, str] = ("call_iv_2nd_up", "low")
    CALL_IV_2ND_UP_CLOSE: tuple[str, str] = ("call_iv_2nd_up", "close")

    PUT_IV_2ND_DOWN_OPEN: tuple[str, str] = ("put_iv_2nd_down", "open")
    PUT_IV_2ND_DOWN_HIGH: tuple[str, str] = ("put_iv_2nd_down", "high")
    PUT_IV_2ND_DOWN_LOW: tuple[str, str] = ("put_iv_2nd_down", "low")
    PUT_IV_2ND_DOWN_CLOSE: tuple[str, str] = ("put_iv_2nd_down", "close")

    CALL_OPENINT_2ND_UP_OPEN: tuple[str, str] = ("call_openint_2nd_up", "open")
    CALL_OPENINT_2ND_UP_HIGH: tuple[str, str] = ("call_openint_2nd_up", "high")
    CALL_OPENINT_2ND_UP_LOW: tuple[str, str] = ("call_openint_2nd_up", "low")
    CALL_OPENINT_2ND_UP_CLOSE: tuple[str, str] = ("call_openint_2nd_up", "close")

    PUT_OPENINT_2ND_DOWN_OPEN: tuple[str, str] = ("put_openint_2nd_down", "open")
    PUT_OPENINT_2ND_DOWN_HIGH: tuple[str, str] = ("put_openint_2nd_down", "high")
    PUT_OPENINT_2ND_DOWN_LOW: tuple[str, str] = ("put_openint_2nd_down", "low")
    PUT_OPENINT_2ND_DOWN_CLOSE: tuple[str, str] = ("put_openint_2nd_down", "close")

    M0S_TOP_TX_201_STRIKE_PRICE_OPEN: tuple[str, str] = (
        "m0s_top_tx_201_strike_price",
        "open",
    )
    M0S_TOP_TX_201_STRIKE_PRICE_HIGH: tuple[str, str] = (
        "m0s_top_tx_201_strike_price",
        "high",
    )
    M0S_TOP_TX_201_STRIKE_PRICE_LOW: tuple[str, str] = (
        "m0s_top_tx_201_strike_price",
        "low",
    )
    M0S_TOP_TX_201_STRIKE_PRICE_CLOSE: tuple[str, str] = (
        "m0s_top_tx_201_strike_price",
        "close",
    )
    M0S_TOP_TX_301_STRIKE_PRICE_OPEN: tuple[str, str] = (
        "m0s_top_tx_301_strike_price",
        "open",
    )
    M0S_TOP_TX_301_STRIKE_PRICE_HIGH: tuple[str, str] = (
        "m0s_top_tx_301_strike_price",
        "high",
    )
    M0S_TOP_TX_301_STRIKE_PRICE_LOW: tuple[str, str] = (
        "m0s_top_tx_301_strike_price",
        "low",
    )
    M0S_TOP_TX_301_STRIKE_PRICE_CLOSE: tuple[str, str] = (
        "m0s_top_tx_301_strike_price",
        "close",
    )

    VKOSPI200_OPEN: tuple[str, str] = ("vkospi200", "open")
    VKOSPI200_HIGH: tuple[str, str] = ("vkospi200", "high")
    VKOSPI200_LOW: tuple[str, str] = ("vkospi200", "low")
    VKOSPI200_REAL_CLOSE: tuple[str, str] = ("vkospi200_real", "close")

    BOND_167_OPEN: tuple[str, str] = ("167_price", "open")
    BOND_167_HIGH: tuple[str, str] = ("167_price", "high")
    BOND_167_LOW: tuple[str, str] = ("167_price", "low")
    BOND_167_CLOSE: tuple[str, str] = ("167_price", "close")

    BOND_165_OPEN: tuple[str, str] = ("165_price", "open")
    BOND_165_HIGH: tuple[str, str] = ("165_price", "high")
    BOND_165_LOW: tuple[str, str] = ("165_price", "low")
    BOND_165_CLOSE: tuple[str, str] = ("165_price", "close")

    USD_PRICE_OPEN: tuple[str, str] = ("usd_price", "open")
    USD_PRICE_HIGH: tuple[str, str] = ("usd_price", "high")
    USD_PRICE_LOW: tuple[str, str] = ("usd_price", "low")
    USD_PRICE_CLOSE: tuple[str, str] = ("usd_price", "close")

    BUY_ORDER_1_QTY_CLOSE: tuple[str, str] = ("buy_order_1", "close")
    BUY_ORDER_2_QTY_CLOSE: tuple[str, str] = ("buy_order_2", "close")
    BUY_ORDER_3_QTY_CLOSE: tuple[str, str] = ("buy_order_3", "close")
    BUY_ORDER_4_QTY_CLOSE: tuple[str, str] = ("buy_order_4", "close")
    BUY_ORDER_5_QTY_CLOSE: tuple[str, str] = ("buy_order_5", "close")

    SELL_ORDER_1_QTY_CLOSE: tuple[str, str] = ("sell_order_1", "close")
    SELL_ORDER_2_QTY_CLOSE: tuple[str, str] = ("sell_order_2", "close")
    SELL_ORDER_3_QTY_CLOSE: tuple[str, str] = ("sell_order_3", "close")
    SELL_ORDER_4_QTY_CLOSE: tuple[str, str] = ("sell_order_4", "close")
    SELL_ORDER_5_QTY_CLOSE: tuple[str, str] = ("sell_order_5", "close")

    BUY_ORDER_1_QTY_VOLUME: tuple[str, str] = ("buy_order_1", "volume")
    BUY_ORDER_2_QTY_VOLUME: tuple[str, str] = ("buy_order_2", "volume")
    BUY_ORDER_3_QTY_VOLUME: tuple[str, str] = ("buy_order_3", "volume")
    BUY_ORDER_4_QTY_VOLUME: tuple[str, str] = ("buy_order_4", "volume")
    BUY_ORDER_5_QTY_VOLUME: tuple[str, str] = ("buy_order_5", "volume")

    SELL_ORDER_1_QTY_VOLUME: tuple[str, str] = ("sell_order_1", "volume")
    SELL_ORDER_2_QTY_VOLUME: tuple[str, str] = ("sell_order_2", "volume")
    SELL_ORDER_3_QTY_VOLUME: tuple[str, str] = ("sell_order_3", "volume")
    SELL_ORDER_4_QTY_VOLUME: tuple[str, str] = ("sell_order_4", "volume")
    SELL_ORDER_5_QTY_VOLUME: tuple[str, str] = ("sell_order_5", "volume")

    BUY_ORDER_1_PRICE_CLOSE: tuple[str, str] = ("buy_order_1_price", "close")
    BUY_ORDER_2_PRICE_CLOSE: tuple[str, str] = ("buy_order_2_price", "close")
    BUY_ORDER_3_PRICE_CLOSE: tuple[str, str] = ("buy_order_3_price", "close")
    BUY_ORDER_4_PRICE_CLOSE: tuple[str, str] = ("buy_order_4_price", "close")
    BUY_ORDER_5_PRICE_CLOSE: tuple[str, str] = ("buy_order_5_price", "close")

    SELL_ORDER_1_PRICE_CLOSE: tuple[str, str] = ("sell_order_1_price", "close")
    SELL_ORDER_2_PRICE_CLOSE: tuple[str, str] = ("sell_order_2_price", "close")
    SELL_ORDER_3_PRICE_CLOSE: tuple[str, str] = ("sell_order_3_price", "close")
    SELL_ORDER_4_PRICE_CLOSE: tuple[str, str] = ("sell_order_4_price", "close")
    SELL_ORDER_5_PRICE_CLOSE: tuple[str, str] = ("sell_order_5_price", "close")

    # Newly added columns for data processed in Dataset.py
    ADJ_CALL_1_PRICE_OPEN: tuple[str, str] = ("adj_call_1_price", "open")
    ADJ_CALL_1_PRICE_HIGH: tuple[str, str] = ("adj_call_1_price", "high")
    ADJ_CALL_1_PRICE_LOW: tuple[str, str] = ("adj_call_1_price", "low")
    ADJ_CALL_1_PRICE_CLOSE: tuple[str, str] = ("adj_call_1_price", "close")

    ADJ_CALL_2_PRICE_OPEN: tuple[str, str] = ("adj_call_2_price", "open")
    ADJ_CALL_2_PRICE_HIGH: tuple[str, str] = ("adj_call_2_price", "high")
    ADJ_CALL_2_PRICE_LOW: tuple[str, str] = ("adj_call_2_price", "low")
    ADJ_CALL_2_PRICE_CLOSE: tuple[str, str] = ("adj_call_2_price", "close")

    ADJ_CALL_3_PRICE_OPEN: tuple[str, str] = ("adj_call_3_price", "open")
    ADJ_CALL_3_PRICE_HIGH: tuple[str, str] = ("adj_call_3_price", "high")
    ADJ_CALL_3_PRICE_LOW: tuple[str, str] = ("adj_call_3_price", "low")
    ADJ_CALL_3_PRICE_CLOSE: tuple[str, str] = ("adj_call_3_price", "close")

    ADJ_CALL_4_PRICE_OPEN: tuple[str, str] = ("adj_call_4_price", "open")
    ADJ_CALL_4_PRICE_HIGH: tuple[str, str] = ("adj_call_4_price", "high")
    ADJ_CALL_4_PRICE_LOW: tuple[str, str] = ("adj_call_4_price", "low")
    ADJ_CALL_4_PRICE_CLOSE: tuple[str, str] = ("adj_call_4_price", "close")

    ADJ_CALL_5_PRICE_OPEN: tuple[str, str] = ("adj_call_5_price", "open")
    ADJ_CALL_5_PRICE_HIGH: tuple[str, str] = ("adj_call_5_price", "high")
    ADJ_CALL_5_PRICE_LOW: tuple[str, str] = ("adj_call_5_price", "low")
    ADJ_CALL_5_PRICE_CLOSE: tuple[str, str] = ("adj_call_5_price", "close")

    ADJ_PUT_1_PRICE_OPEN: tuple[str, str] = ("adj_put_1_price", "open")
    ADJ_PUT_1_PRICE_HIGH: tuple[str, str] = ("adj_put_1_price", "high")
    ADJ_PUT_1_PRICE_LOW: tuple[str, str] = ("adj_put_1_price", "low")
    ADJ_PUT_1_PRICE_CLOSE: tuple[str, str] = ("adj_put_1_price", "close")

    ADJ_PUT_2_PRICE_OPEN: tuple[str, str] = ("adj_put_2_price", "open")
    ADJ_PUT_2_PRICE_HIGH: tuple[str, str] = ("adj_put_2_price", "high")
    ADJ_PUT_2_PRICE_LOW: tuple[str, str] = ("adj_put_2_price", "low")
    ADJ_PUT_2_PRICE_CLOSE: tuple[str, str] = ("adj_put_2_price", "close")

    ADJ_PUT_3_PRICE_OPEN: tuple[str, str] = ("adj_put_3_price", "open")
    ADJ_PUT_3_PRICE_HIGH: tuple[str, str] = ("adj_put_3_price", "high")
    ADJ_PUT_3_PRICE_LOW: tuple[str, str] = ("adj_put_3_price", "low")
    ADJ_PUT_3_PRICE_CLOSE: tuple[str, str] = ("adj_put_3_price", "close")

    ADJ_PUT_4_PRICE_OPEN: tuple[str, str] = ("adj_put_4_price", "open")
    ADJ_PUT_4_PRICE_HIGH: tuple[str, str] = ("adj_put_4_price", "high")
    ADJ_PUT_4_PRICE_LOW: tuple[str, str] = ("adj_put_4_price", "low")
    ADJ_PUT_4_PRICE_CLOSE: tuple[str, str] = ("adj_put_4_price", "close")

    ADJ_PUT_5_PRICE_OPEN: tuple[str, str] = ("adj_put_5_price", "open")
    ADJ_PUT_5_PRICE_HIGH: tuple[str, str] = ("adj_put_5_price", "high")
    ADJ_PUT_5_PRICE_LOW: tuple[str, str] = ("adj_put_5_price", "low")
    ADJ_PUT_5_PRICE_CLOSE: tuple[str, str] = ("adj_put_5_price", "close")

    ADJ_CALL_1_AMT: tuple[str, str] = ("adj_call_1_amt", "volume")
    ADJ_CALL_2_AMT: tuple[str, str] = ("adj_call_2_amt", "volume")
    ADJ_CALL_3_AMT: tuple[str, str] = ("adj_call_3_amt", "volume")
    ADJ_CALL_4_AMT: tuple[str, str] = ("adj_call_4_amt", "volume")
    ADJ_CALL_5_AMT: tuple[str, str] = ("adj_call_5_amt", "volume")

    ADJ_PUT_1_AMT: tuple[str, str] = ("adj_put_1_amt", "volume")
    ADJ_PUT_2_AMT: tuple[str, str] = ("adj_put_2_amt", "volume")
    ADJ_PUT_3_AMT: tuple[str, str] = ("adj_put_3_amt", "volume")
    ADJ_PUT_4_AMT: tuple[str, str] = ("adj_put_4_amt", "volume")
    ADJ_PUT_5_AMT: tuple[str, str] = ("adj_put_5_amt", "volume")

    ADJ_CALL_1_VOL: tuple[str, str] = ("adj_call_1_vol", "volume")
    ADJ_CALL_2_VOL: tuple[str, str] = ("adj_call_2_vol", "volume")
    ADJ_CALL_3_VOL: tuple[str, str] = ("adj_call_3_vol", "volume")
    ADJ_CALL_4_VOL: tuple[str, str] = ("adj_call_4_vol", "volume")
    ADJ_CALL_5_VOL: tuple[str, str] = ("adj_call_5_vol", "volume")

    ADJ_PUT_1_VOL: tuple[str, str] = ("adj_put_1_vol", "volume")
    ADJ_PUT_2_VOL: tuple[str, str] = ("adj_put_2_vol", "volume")
    ADJ_PUT_3_VOL: tuple[str, str] = ("adj_put_3_vol", "volume")
    ADJ_PUT_4_VOL: tuple[str, str] = ("adj_put_4_vol", "volume")
    ADJ_PUT_5_VOL: tuple[str, str] = ("adj_put_5_vol", "volume")

    ADJ_CALL_1_OPENINT_OPEN: tuple[str, str] = ("adj_call_1_oi", "open")
    ADJ_CALL_1_OPENINT_HIGH: tuple[str, str] = ("adj_call_1_oi", "high")
    ADJ_CALL_1_OPENINT_LOW: tuple[str, str] = ("adj_call_1_oi", "low")
    ADJ_CALL_1_OPENINT_CLOSE: tuple[str, str] = ("adj_call_1_oi", "close")

    ADJ_CALL_2_OPENINT_OPEN: tuple[str, str] = ("adj_call_2_oi", "open")
    ADJ_CALL_2_OPENINT_HIGH: tuple[str, str] = ("adj_call_2_oi", "high")
    ADJ_CALL_2_OPENINT_LOW: tuple[str, str] = ("adj_call_2_oi", "low")
    ADJ_CALL_2_OPENINT_CLOSE: tuple[str, str] = ("adj_call_2_oi", "close")

    ADJ_CALL_3_OPENINT_OPEN: tuple[str, str] = ("adj_call_3_oi", "open")
    ADJ_CALL_3_OPENINT_HIGH: tuple[str, str] = ("adj_call_3_oi", "high")
    ADJ_CALL_3_OPENINT_LOW: tuple[str, str] = ("adj_call_3_oi", "low")
    ADJ_CALL_3_OPENINT_CLOSE: tuple[str, str] = ("adj_call_3_oi", "close")

    ADJ_CALL_4_OPENINT_OPEN: tuple[str, str] = ("adj_call_4_oi", "open")
    ADJ_CALL_4_OPENINT_HIGH: tuple[str, str] = ("adj_call_4_oi", "high")
    ADJ_CALL_4_OPENINT_LOW: tuple[str, str] = ("adj_call_4_oi", "low")
    ADJ_CALL_4_OPENINT_CLOSE: tuple[str, str] = ("adj_call_4_oi", "close")

    ADJ_CALL_5_OPENINT_OPEN: tuple[str, str] = ("adj_call_5_oi", "open")
    ADJ_CALL_5_OPENINT_HIGH: tuple[str, str] = ("adj_call_5_oi", "high")
    ADJ_CALL_5_OPENINT_LOW: tuple[str, str] = ("adj_call_5_oi", "low")
    ADJ_CALL_5_OPENINT_CLOSE: tuple[str, str] = ("adj_call_5_oi", "close")

    ADJ_PUT_1_OPENINT_OPEN: tuple[str, str] = ("adj_put_1_oi", "open")
    ADJ_PUT_1_OPENINT_HIGH: tuple[str, str] = ("adj_put_1_oi", "high")
    ADJ_PUT_1_OPENINT_LOW: tuple[str, str] = ("adj_put_1_oi", "low")
    ADJ_PUT_1_OPENINT_CLOSE: tuple[str, str] = ("adj_put_1_oi", "close")

    ADJ_PUT_2_OPENINT_OPEN: tuple[str, str] = ("adj_put_2_oi", "open")
    ADJ_PUT_2_OPENINT_HIGH: tuple[str, str] = ("adj_put_2_oi", "high")
    ADJ_PUT_2_OPENINT_LOW: tuple[str, str] = ("adj_put_2_oi", "low")
    ADJ_PUT_2_OPENINT_CLOSE: tuple[str, str] = ("adj_put_2_oi", "close")

    ADJ_PUT_3_OPENINT_OPEN: tuple[str, str] = ("adj_put_3_oi", "open")
    ADJ_PUT_3_OPENINT_HIGH: tuple[str, str] = ("adj_put_3_oi", "high")
    ADJ_PUT_3_OPENINT_LOW: tuple[str, str] = ("adj_put_3_oi", "low")
    ADJ_PUT_3_OPENINT_CLOSE: tuple[str, str] = ("adj_put_3_oi", "close")

    ADJ_PUT_4_OPENINT_OPEN: tuple[str, str] = ("adj_put_4_oi", "open")
    ADJ_PUT_4_OPENINT_HIGH: tuple[str, str] = ("adj_put_4_oi", "high")
    ADJ_PUT_4_OPENINT_LOW: tuple[str, str] = ("adj_put_4_oi", "low")
    ADJ_PUT_4_OPENINT_CLOSE: tuple[str, str] = ("adj_put_4_oi", "close")

    ADJ_PUT_5_OPENINT_OPEN: tuple[str, str] = ("adj_put_5_oi", "open")
    ADJ_PUT_5_OPENINT_HIGH: tuple[str, str] = ("adj_put_5_oi", "high")
    ADJ_PUT_5_OPENINT_LOW: tuple[str, str] = ("adj_put_5_oi", "low")
    ADJ_PUT_5_OPENINT_CLOSE: tuple[str, str] = ("adj_put_5_oi", "close")


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
RESAMPLE_RULE_TO_MIN = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
}
