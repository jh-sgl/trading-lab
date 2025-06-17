import torch
import pandas as pd
import numpy as np
from scipy import stats
from alphasearch_naive.utils import path_cfg
from omegaconf import DictConfig
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use("ggplot")

UWON = 100_000_000


def get_margin_rate():
    margin_rate = pd.read_csv(path_cfg.margin_rate_path)
    margin_rate = margin_rate.set_index("date")
    margin_rate.index = pd.to_datetime(margin_rate.index)
    margin_rate = margin_rate.sort_index()
    return margin_rate


def modify_volume(volume, min_size=1, round_decimals=0):
    v = abs(volume)
    v = np.clip(v, min_size, float("inf"))
    v = np.round(v, round_decimals)
    return v


def get_betting_size(equity, risk, loss, tick_value=250_000):
    volume = (equity * risk) / (loss * tick_value)
    return volume


def convert_to_trade(
    trade_df,
    sizing,
    method="wsum",
    side="both",
    threshold=0,
    slippage=0.05,
    balance=1_000_000_000,
    commission=0.000029,
    risk=0.05,
    vol_coef=100,
    vol_clip_min=0.2,
    tick_value=250_000,
    **kwargs,
):
    # 여기서 tick value 는 실제 min_tick_size 의 가치가 아니라, 표시된 값 1당 실제 얼마의 가치를 가지는 지 임.
    # tick_value 가 25만원 default 인 것은 코스피지수 1당 실제 25만원이라서 그런것임.
    df = trade_df.copy()
    assert method in ["wsum", "argmax"]
    assert side in ["long", "short", "both"]
    assert sizing in ["vol", "vol_clip", "fixed"]

    # handle method
    if method == "argmax":
        decision_ = df["argmax_decision"]
    elif method == "wsum":
        decision_ = df["decision"].copy()
        decision_[abs(decision_) < threshold] = 0
    else:
        raise RuntimeError()

    decision = decision_.copy()
    if side == "long":
        decision[decision < 0] = 0
    elif side == "short":
        decision[decision > 0] = 0

    vol = df["vol"]
    if sizing == "vol":
        size = get_betting_size(balance, risk, vol * vol_coef, tick_value=tick_value)
    elif sizing == "vol_clip":
        vol_clipped = np.clip(vol, a_min=vol_clip_min, a_max=10)
        size = get_betting_size(
            balance, risk, vol_clipped * vol_coef, tick_value=tick_value
        )
    elif sizing == "fixed":
        size = get_betting_size(
            balance,
            risk,
            np.ones_like(vol.values) * 0.15 * vol_coef,
            tick_value=tick_value,
        )

    position_size = modify_volume(decision * size) * np.sign(decision)
    cost = (abs(position_size) * slippage * 2) + (
        abs(position_size) * df["price"] * commission * 2
    )
    profit = position_size * df["rt"] - cost

    df["profit"] = profit
    df["profit_won"] = profit * tick_value
    df["profit_uwon"] = profit * tick_value / UWON
    df["qty"] = position_size
    df["cost"] = cost

    return df


def calc_margin(profit_df, margin_rate, balance):

    max_naked_balance = profit_df.groupby(profit_df.index.date).apply(
        lambda x: (x.qty.cumsum() * x.price * 250_000 / UWON).abs().max()
    )
    max_position = profit_df.groupby(profit_df.index.date).apply(
        lambda x: (x.qty.cumsum()).abs().max()
    )

    max_naked_balance.index = pd.to_datetime(max_naked_balance.index)
    max_naked_balance = pd.DataFrame(max_naked_balance, columns=["max_naked_balance"])
    max_naked_balance["max_position"] = max_position.values

    margin = pd.merge_asof(
        max_naked_balance.reset_index(),
        margin_rate.reset_index(),
        left_on="index",
        right_on="date",
        direction="backward",
    )

    margin["max_margin"] = margin["max_naked_balance"] * margin["initial_margin_rate"]
    margin["margin_ratio"] = margin["max_margin"] * UWON / balance
    margin = margin[
        [
            "index",
            "max_margin",
            "max_position",
            "margin_ratio",
            "max_naked_balance",
            "initial_margin_rate",
            "trade_margin_rate",
        ]
    ]

    margin = margin.set_index("index")
    margin.index = margin.index.rename("datetime")
    return margin


def cut_off_margin(profit_df, margin_df, margin_threshold):

    def cutoff_daily(df, threshold):
        items = []
        date = df.index.date[0]
        init_margin_rate = (
            margin_df[margin_df.index.date == date].iloc[0].initial_margin_rate
        )
        cum_qty = 0
        for i in range(len(df)):
            item = df.iloc[i]

            expected_margin = (
                abs((cum_qty + item.qty) * item.price * 250_000 / UWON)
                * init_margin_rate
            )

            if expected_margin >= threshold:
                continue
            else:
                items.append(item)
                cum_qty += item.qty

        return pd.DataFrame(items)

    profit_df = (
        profit_df.groupby(profit_df.index.date, group_keys=False)
        .apply(lambda x: cutoff_daily(x, margin_threshold))
        .copy()
    )
    return profit_df


def calculate_drawdown(df):
    df["cumulative_profit"] = df["profit_uwon"].cumsum()
    # 현재까지의 최대 누적 수익 계산
    df["peak"] = df["cumulative_profit"].cummax()
    # Drawdown 계산 (원화 단위)
    df["drawdown"] = df["peak"] - df["cumulative_profit"]
    # Drawdown 시작점, 종료점 식별
    df["drawdown_start"] = (df["drawdown"] != 0) & (df["drawdown"].shift(1) == 0)
    df["drawdown_end"] = (df["drawdown"] == 0) & (df["drawdown"].shift(1) != 0)
    return df


def calculate_drawdown_compound(df):
    # 누적 수익률 계산 (1을 더해 복리 효과 반영)
    df["cumulative_return"] = (1 + df["profit"]).cumprod() - 1
    # 현재까지의 최대 누적 수익률 계산
    df["peak"] = df["cumulative_return"].cummax()
    # Drawdown 계산 (퍼센트 단위)
    df["drawdown"] = (df["peak"] - df["cumulative_return"]) / (1 + df["peak"])
    # Drawdown 시작점 식별
    df["drawdown_start"] = (df["drawdown"] != 0) & (df["drawdown"].shift(1) == 0)
    # Drawdown 종료점 식별
    df["drawdown_end"] = (df["drawdown"] == 0) & (df["drawdown"].shift(1) != 0)
    return df


def find_top_drawdowns(df, top_n=10):
    drawdown_periods = []
    current_start = None

    for idx, row in df.iterrows():
        if row["drawdown_start"]:
            current_start = idx
        elif row["drawdown_end"] and current_start is not None:
            drawdown_periods.append(
                {
                    "start": current_start,
                    "end": idx,
                    "max_drawdown": df.loc[current_start:idx, "drawdown"].max(),
                    "duration": (idx - current_start).days,
                }
            )
            current_start = None

    # 마지막 drawdown이 아직 진행 중인 경우
    if current_start is not None:
        drawdown_periods.append(
            {
                "start": current_start,
                "end": df.index[-1],
                "max_drawdown": df.loc[current_start:, "drawdown"].max(),
                "duration": (df.index[-1] - current_start).days,
            }
        )

    # drawdown 크기에 따라 정렬하고 top N 선택
    top_drawdowns = sorted(
        drawdown_periods, key=lambda x: x["max_drawdown"], reverse=True
    )[:top_n]
    top_drawdowns = pd.DataFrame(top_drawdowns)
    top_drawdowns.start = top_drawdowns.start.dt.date
    top_drawdowns.end = top_drawdowns.end.dt.date

    top_drawdowns.max_drawdown = top_drawdowns.max_drawdown.round(2)
    return pd.DataFrame(top_drawdowns)


def calc_stats(trade_df, profit_df, margin_rate, years, balance):

    profit_df = profit_df[profit_df.index.year.isin(years)].copy()

    margin_df = calc_margin(profit_df, margin_rate, balance=balance)

    max_margin = margin_df.max_margin.max()
    adjusted_balance = max_margin * 1.2

    profit_df = calculate_drawdown(profit_df)
    top_drawdowns = find_top_drawdowns(profit_df)
    max_drawdown = top_drawdowns.iloc[0].max_drawdown

    total_profit = profit_df.profit_uwon.cumsum().iloc[-1]

    num_days = len(np.unique(trade_df[trade_df.index.year.isin(years)].index.date))
    annual_profit = total_profit / num_days * 252

    o_ratio = annual_profit / max_drawdown

    daily_profit_df = profit_df.groupby(profit_df.index.date, group_keys=False).apply(
        lambda x: x.profit.sum()
    )
    sharpe = daily_profit_df.mean() / daily_profit_df.std() * np.sqrt(252)

    skew = stats.skew(daily_profit_df)
    kurtosis = stats.kurtosis(daily_profit_df)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.arange(len(daily_profit_df)), daily_profit_df.cumsum().values
    )
    r_square = r_value**2

    profit_df_compound = profit_df.copy()
    pct_return = profit_df["profit_uwon"] / adjusted_balance
    profit_df_compound["profit"] = pct_return
    compound_total_return = (1 + pct_return).prod()
    compound_daily_return = ((1 + compound_total_return) ** (1 / num_days)) - 1
    compound_yearly_return = (1 + compound_daily_return) ** 252 - 1
    compound_dd = calculate_drawdown_compound(profit_df_compound)
    compound_mdd = compound_dd["drawdown"].max()

    info_dict = {
        "dd": top_drawdowns,
        "stat": pd.Series(
            {
                "max_margin": max_margin,
                "adjusted_balance": adjusted_balance,
                "mdd": max_drawdown,
                "total_profit": total_profit,
                "annual_profit": annual_profit,
                "o_ratio": o_ratio,
                "sharpe": sharpe,
                "r_square": r_square,
                "kurtosis": kurtosis,
                "skew": skew,
                "num_days": num_days,
                "compound_total_return": compound_total_return,
                "compound_yearly_return": compound_yearly_return,
                "compound_mdd": compound_mdd,
            }
        ),
    }
    return info_dict


def plot(trade_df, profit_df, margin_rate, balance, years, meta_info=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.cla()
    plt.clf()
    profit_df = profit_df[profit_df.index.year.isin(years)].copy()
    trade_df = trade_df[
        (profit_df.index.date.min() <= trade_df.index.date)
        & (trade_df.index.date <= profit_df.index.date.max())
    ].copy()
    margin_df = calc_margin(profit_df, margin_rate, balance)
    fig = plt.figure(figsize=(20, 10))

    col_offset = 1 if meta_info is not None else 0
    num_cols = 3 + col_offset
    gs = GridSpec(2, num_cols, figure=fig, height_ratios=[1, 0.8])

    ax1 = fig.add_subplot(gs[0, 0 + col_offset : 2 + col_offset])
    ax2 = fig.add_subplot(gs[1, 0 + col_offset : 1 + col_offset])
    ax2b = fig.add_subplot(gs[1, 1 + col_offset : 2 + col_offset])

    ax3 = fig.add_subplot(gs[0, 2 + col_offset])
    ax3b = fig.add_subplot(gs[1, 2 + col_offset])

    if meta_info is not None:
        ax0 = fig.add_subplot(gs[0:2, 0])
        table = ax0.table(
            cellText=meta_info,
            colLabels=["Name", "Value"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)
        ax0.axis("off")
        ax0.set_title("Meta Info", pad=0)

    # ---------------------- ax1 (balance 부분) ----------------------
    # 왼쪽 y축에 balance를 점으로 그리기
    from_, to_ = str(profit_df.index.date.min()), str(profit_df.index.date.max())
    simple_balance = profit_df.profit_uwon.cumsum()
    long_balance = profit_df[profit_df.decision > 0].profit_uwon.cumsum()
    short_balance = profit_df[profit_df.decision < 0].profit_uwon.cumsum()
    # long_balance = long_balance * (simple_balance.iloc[-1] / long_balance.iloc[-1])
    # short_balance = short_balance * (simple_balance.iloc[-1] / short_balance.iloc[-1])
    # ax1.scatter(simple_balance.index, simple_balance.values, color='indianred', s=1, alpha=1, label='Balance')
    ax1.plot(
        simple_balance.index,
        simple_balance.values,
        color="indianred",
        # s=1,
        alpha=1,
        label="Balance",
    )
    ax1.plot(
        long_balance.index,
        long_balance.values,
        color="lightsalmon",
        # s=1,
        alpha=0.5,
        label="Long Only Balance",
    )
    ax1.plot(
        short_balance.index,
        short_balance.values,
        color="cornflowerblue",
        # s=1,
        alpha=0.5,
        label="Short Only Balance",
    )
    # 오른쪽 y축 생성
    ax1_right = ax1.twinx()

    # 오른쪽 y축에 position size 그리기
    position_size = profit_df.groupby(profit_df.index.date, group_keys=False).apply(
        lambda x: np.sign(x.qty.cumsum().iloc[-1]) * x.qty.cumsum().abs().max()
    )
    ax1_right.bar(
        x=position_size.index,
        height=position_size,
        # width=3,
        color="dimgray",
        alpha=0.5,
        label="Position Size",
    )

    # 범례 추가
    ax1.legend(loc="upper left")
    ax1_right.legend(loc="lower right")

    # 축 레이블 추가
    ax1.set_ylabel("Balance")
    ax1_right.set_ylabel("Position Size")

    # 제목 추가
    ax1.set_title(f"Balance and Position Size Over Time")

    # ---------------------- ax2 (position hist) ----------------------
    position_size[position_size != 0].hist(
        bins=200, ax=ax2, color="dimgray", alpha=0.7, label="daily max position"
    )
    ax2.legend(loc="upper left")
    ax2.set_title("Max Position Size Histogram")
    market_exposure = len(position_size[position_size != 0]) / len(
        np.unique(trade_df[trade_df.index.year.isin(years)].index.date)
    )

    # ax2b
    margin_rank = margin_df.sort_values("max_margin", ascending=False).iloc[:5][
        ["max_margin", "max_position", "initial_margin_rate"]
    ]
    position_rank = margin_df.sort_values("max_position", ascending=False).iloc[:5][
        ["max_margin", "max_position", "initial_margin_rate"]
    ]
    margin_position_rank = pd.concat([margin_rank, position_rank], axis=0)
    margin_position_rank.index = margin_position_rank.index.date
    margin_position_rank = margin_position_rank.reset_index()
    table_margin = ax2b.table(
        cellText=margin_position_rank.round(3).values,
        colLabels=margin_position_rank.columns,
        cellLoc="center",
        loc="center",
    )
    table_margin.auto_set_font_size(False)
    table_margin.set_fontsize(7)
    table_margin.scale(1, 1.5)
    ax2b.axis("off")
    ax2b.set_title("Max Position size")

    # ---------------------- ax3 (stats) ----------------------
    info_dict = calc_stats(trade_df, profit_df, margin_rate, years, balance)
    info_stat = info_dict["stat"]
    info_dd = info_dict["dd"]
    # 메트릭 테이블 추가
    table_data = [
        [
            "Total Return",
            f"{info_stat['total_profit']:.2f} Uwon / {info_stat['total_profit']/info_stat['adjusted_balance'] * 100:.3f} % ",
            "Simple Interest",
        ],
        [
            "Anual Return",
            f"{info_stat['annual_profit']:.2f} Uwon / {info_stat['annual_profit']/info_stat['adjusted_balance'] * 100:.3f} % ",
            "Simple Interest",
        ],
        [
            "Realistic Capital",
            f"{info_stat['adjusted_balance']:.2f} Uwon",
            "(1.2X of max margin)",
        ],
        [
            "MDD Capital",
            f"{info_stat['max_margin'] + info_stat['mdd']:.2f} Uwon",
            "(max margin + mdd)",
        ],
        [
            "Max Drawdown",
            f"{info_stat['mdd']:.2f} Uwon / {info_stat['mdd']/info_stat['adjusted_balance'] * 100:.3f} %",
            "mdd / Realistic Capital",
        ],
        [
            "Max Position Size",
            f"{position_rank.iloc[0].max_position:.2f} @ {position_rank.index.date[0]}",
            "",
        ],
        [
            "Max Margin",
            f"{margin_rank.iloc[0].max_margin:.2f} Uwon @ {margin_rank.index.date[0]}",
            "",
        ],
        [
            "Market Exposure",
            f"{market_exposure*100:.2f}",
            "(Trading days)/(Total days)",
        ],
        ["O-Ratio", f"{info_stat['o_ratio']:.2f}", ""],
        ["Sharpe-Ratio", f"{info_stat['sharpe']:.2f}", ""],
        ["R Square", f"{info_stat['r_square']:.3f}", ""],
        ["skew", f"{info_stat['skew']:.3f}", ""],
        ["kurtosis", f"{info_stat['kurtosis']:.3f}", ""],
        [
            "Total Return (Compound)",
            f"{info_stat['compound_total_return'] * 100:.2f} % ",
            "Compound Interest",
        ],
        [
            "Anual Return (Compound)",
            f"{info_stat['compound_yearly_return'] * 100:.2f} % ",
            "Compound Interest",
        ],
        [
            "Max Drawdown (Compound)",
            f"{info_stat['compound_mdd'] * 100:.2f} % ",
            "Compound Interest",
        ],
        [
            "O-Ratio (Compound)",
            f"{info_stat['compound_yearly_return'] / info_stat['compound_mdd']:.2f} ",
            "Compound Interest",
        ],
    ]
    table = ax3.table(
        cellText=table_data,
        colLabels=["Metric", "Value", "Description"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    ax3.axis("off")
    ax3.set_title("Performance Metrics", pad=20)

    # -------------------- ax3b (drawdowns)
    table_dd = ax3b.table(
        cellText=info_dd.round(3).values,
        colLabels=info_dd.columns,
        cellLoc="center",
        loc="center",
    )
    table_dd.auto_set_font_size(False)
    table_dd.set_fontsize(8)
    table_dd.scale(1, 1.5)
    ax3b.axis("off")
    ax3b.set_title("Worst Draw Downs")

    # ---------------------- ax4 (hist 부분) ----------------------
    # 왼쪽 y축에 balance를 점으로 그리기

    plt.suptitle(f"{from_} ~ {to_}", fontsize=16, y=0.98)
    plt.tight_layout()
    return fig, info_stat
