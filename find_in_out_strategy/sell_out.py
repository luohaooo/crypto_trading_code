"""Utilities for scanning the 15-minute cache for specific entry/exit setups."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# Allow overriding the cache directory via env variable so backtests stay configurable
FIFTEEN_CACHE_DIR = Path(
    os.getenv("PICKLE_15M_CACHE", "/home/craz/crypto/crypto-data/pickle_15m_cache")
)
MINUTE_CACHE_DIR = Path(
    os.getenv("PICKLE_1M_CACHE", "/home/craz/crypto/crypto-data/pickle_month_cache")
)


def _normalize_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Convert user input into a timezone-naive pandas Timestamp."""
    ts = pd.Timestamp(value)
    # Strip tz info to match cached data
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _month_periods(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Period]:
    """Yield each month (Period) between start and end inclusive."""
    start_period = start.to_period("M")
    end_period = end.to_period("M")
    return pd.period_range(start_period, end_period, freq="M")


def load_15m_data(
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    data_dir: Path = FIFTEEN_CACHE_DIR,
) -> pd.DataFrame:
    """Load 15-minute OHLCV rows between the requested timestamps.

    The data files follow the naming pattern ``usdt_data_YYYY-MM.pkl`` and contain a
    MultiIndex ``(open_time, symbol)``. Only rows inside the requested time window
    are returned.
    """
    start_ts = _normalize_timestamp(start_time)
    end_ts = _normalize_timestamp(end_time)
    if end_ts <= start_ts:
        raise ValueError("end_time must be greater than start_time")

    frames: list[pd.DataFrame] = []
    for period in _month_periods(start_ts, end_ts):
        file_name = f"usdt_data_{period.strftime('%Y-%m')}.pkl"
        file_path = data_dir / file_name
        if not file_path.exists():
            # Missing months are expected sometimes, so just skip them.
            continue
        frames.append(pd.read_pickle(file_path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    open_times = df.index.get_level_values("open_time")
    mask = (open_times >= start_ts) & (open_times <= end_ts)
    return df.loc[mask].sort_index()


def load_1m_data(
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    data_dir: Path = MINUTE_CACHE_DIR,
) -> pd.DataFrame:
    """Load 1-minute OHLCV rows between ``start_time`` and ``end_time``."""
    start_ts = _normalize_timestamp(start_time)
    end_ts = _normalize_timestamp(end_time)
    if end_ts <= start_ts:
        raise ValueError("end_time must be greater than start_time")

    frames: list[pd.DataFrame] = []
    for period in _month_periods(start_ts, end_ts):
        file_name = f"usdt_data_{period.strftime('%Y-%m')}.pkl"
        file_path = data_dir / file_name
        if not file_path.exists():
            continue
        frames.append(pd.read_pickle(file_path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    open_times = df.index.get_level_values("open_time")
    mask = (open_times >= start_ts) & (open_times <= end_ts)
    return df.loc[mask].sort_index()


ConditionFn = Callable[[pd.DataFrame], pd.DataFrame]
StopLevelFn = Callable[
    [pd.Series, float, Optional[float], Optional[float], dict],
    tuple[float, float, dict],
]


@dataclass(frozen=True)
class StopParameterConfig:
    """Configuration container for a specific stop/take-profit parameter set."""

    name: str
    determine_stop_fn: StopLevelFn
    metadata: Optional[dict[str, Any]] = None

def condition(
    symbol_df: pd.DataFrame
) -> pd.DataFrame:
    """Return rows where ``volume`` is ``multiplier``× the avg of previous ``window`` rows."""
    # Work on a symbol-specific index (open_time)
    symbol_only = symbol_df.droplevel("symbol")
    volumes = symbol_only["volume"].astype(float)
    open_prices = symbol_only["open"].astype(float)
    close_prices = symbol_only["close"].astype(float)
    high_prices = symbol_only["high"].astype(float)
    low_prices = symbol_only["low"].astype(float)

    volumes_previous_1 = volumes.shift(1)
    volumes_previous_2 = volumes.shift(2)
    volumes_previous_3 = volumes.shift(3)

    rolling_avg_volumes_1_20 = volumes_previous_1.rolling(window=20, min_periods=20).mean()
    rolling_avg_volumes_2_21 = volumes_previous_2.rolling(window=20, min_periods=20).mean()
    rolling_avg_volumes_3_22 = volumes_previous_3.rolling(window=20, min_periods=20).mean()

    volume_trigger = (volumes >= 20 * rolling_avg_volumes_1_20) 

    bullish = close_prices > open_prices
    bullish_streak = bullish 

    mask = volume_trigger & bullish_streak
    mask = mask.reindex(symbol_only.index, fill_value=False)
    return symbol_df[mask.values]

def determine_next_stop_levels(
    last_bar: pd.Series,
    entry_price: float,
    take_profit: Optional[float],
    stop_loss: Optional[float],
    stage: Optional[dict] = None,
    *,
    take_profit_buffer: float,
    stop_loss_buffer: float,
    trailing_factor: float,
) -> tuple[float, float, dict]:
    """Update trailing stop levels using the latest minute close."""
    if stage is None:
        stage = {}

    close_price = float(last_bar["close"])
    high_price = float(last_bar["high"])
    low_price = float(last_bar["low"])

    stage["highest"] = max(stage.get("highest", entry_price), high_price)
    stage["lowest"] = min(stage.get("lowest", entry_price), low_price)
    stage["counter"] = stage.get("counter", 0) + 1

    if take_profit is None:
        take_profit = entry_price * (1 + take_profit_buffer)
    else:
        previous_take_profit = stage.get("previous_take_profit", entry_price)
        take_profit = previous_take_profit - (
            (-stage["lowest"] + previous_take_profit) * trailing_factor
        )

    if stop_loss is None:
        stop_loss = entry_price * (1 - stop_loss_buffer)
    else:
        stop_loss = min(take_profit, close_price * (1 - stop_loss_buffer))

    stage["previous_take_profit"] = take_profit
    stage["previous_stop_loss"] = stop_loss
    return take_profit, stop_loss, stage


def make_stop_level_fn(
    *,
    take_profit_buffer: float,
    stop_loss_buffer: float,
    trailing_factor: float,
) -> StopLevelFn:
    """Create a stop-level function with frozen parameter buffers."""

    def _wrapped(
        last_bar: pd.Series,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        stage: Optional[dict],
    ) -> tuple[float, float, dict]:
        return determine_next_stop_levels(
            last_bar,
            entry_price,
            take_profit,
            stop_loss,
            stage,
            take_profit_buffer=take_profit_buffer,
            stop_loss_buffer=stop_loss_buffer,
            trailing_factor=trailing_factor,
        )

    return _wrapped


def evaluate_stop_trigger(
    last_bar: pd.Series, take_profit: Optional[float], stop_loss: Optional[float]
) -> tuple[Optional[str], Optional[float]]:
    """Check whether the previous minute hit take-profit or stop-loss."""
    high_price = float(last_bar["high"])
    low_price = float(last_bar["low"])

    if take_profit is not None and high_price >= take_profit:
        return "take_profit", take_profit
    if stop_loss is not None and low_price <= stop_loss:
        return "stop_loss", stop_loss
    return None, None


def simulate_minute_trade(
    symbol: str,
    entry_time: pd.Timestamp,
    *,
    determine_stop_fn: StopLevelFn,
    max_minutes: int = 1440 * 7,
) -> Optional[dict]:
    """Simulate per-minute stop logic starting from ``entry_time``."""
    multi_config = StopParameterConfig(name="single", determine_stop_fn=determine_stop_fn)
    multi_result = simulate_minute_trade_multi(
        symbol, entry_time, [multi_config], max_minutes=max_minutes
    )
    if multi_result is None:
        return None
    return multi_result.get(multi_config.name)


def simulate_minute_trade_multi(
    symbol: str,
    entry_time: pd.Timestamp,
    parameter_configs: Sequence[StopParameterConfig],
    *,
    max_minutes: int = 1440 * 7,
) -> Optional[dict[str, dict]]:
    """Simulate per-minute stops for multiple parameter configurations."""
    if not parameter_configs:
        return {}

    minute_data = load_1m_data(entry_time, entry_time + pd.Timedelta(minutes=max_minutes))
    if minute_data.empty or symbol not in minute_data.index.get_level_values("symbol"):
        return None

    symbol_minutes = minute_data.xs(symbol, level="symbol").sort_index()
    trade_minutes = symbol_minutes.loc[entry_time:]
    if trade_minutes.empty:
        return None

    iterator = trade_minutes.iloc[:max_minutes].iterrows()
    try:
        entry_idx, prev_bar = next(iterator)
    except StopIteration:
        return None

    entry_price = float(prev_bar["open"])

    states: dict[str, dict[str, Any]] = {}
    for config in parameter_configs:
        stage: dict = {}
        take_profit, stop_loss, stage = config.determine_stop_fn(
            prev_bar, entry_price, None, None, stage
        )
        states[config.name] = {
            "config": config,
            "stage": stage,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "exit_price": None,
            "exit_reason": None,
            "exit_time": None,
        }

    active_configs = set(states.keys())
    last_idx = entry_idx

    for ts, bar in iterator:
        for name, state in states.items():
            if name not in active_configs:
                continue
            trigger_reason, triggered_price = evaluate_stop_trigger(
                prev_bar, state["take_profit"], state["stop_loss"]
            )
            if trigger_reason:
                state["exit_price"] = triggered_price
                state["exit_reason"] = trigger_reason
                state["exit_time"] = ts
                active_configs.discard(name)
                continue

            take_profit, stop_loss, stage = state["config"].determine_stop_fn(
                prev_bar,
                entry_price,
                state["take_profit"],
                state["stop_loss"],
                state["stage"],
            )
            state["take_profit"] = take_profit
            state["stop_loss"] = stop_loss
            state["stage"] = stage

        prev_bar = bar
        last_idx = ts
        if not active_configs:
            break
    else:
        for name in active_configs:
            state = states[name]
            state["exit_price"] = float(prev_bar["close"])
            state["exit_reason"] = "timeout"
            state["exit_time"] = last_idx
        active_configs.clear()

    trades: dict[str, dict] = {}
    for name, state in states.items():
        trades[name] = {
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": state["exit_time"],
            "exit_price": state["exit_price"],
            "exit_reason": state["exit_reason"],
        }

    return trades


def find_events(
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    condition_fn: ConditionFn,
    *,
    data_dir: Path = FIFTEEN_CACHE_DIR,
) -> list[Tuple[pd.Timestamp, str, pd.Series]]:
    """Scan each symbol and collect rows that satisfy ``condition_fn``."""
    dataset = load_15m_data(start_time, end_time, data_dir=data_dir)
    if dataset.empty:
        print("No data found for the requested range.")
        return []

    results: list[Tuple[pd.Timestamp, str, pd.Series]] = []
    for symbol, symbol_df in dataset.groupby(level="symbol"):
        symbol_df = symbol_df.sort_index(level="open_time")
        matched = condition_fn(symbol_df)
        for (open_time, _), row in matched.iterrows():
            print(
                f"{open_time} - {symbol}: volume={row['volume']:.2f} "
                f"open={row['open']:.4f} close={row['close']:.4f}"
            )
            results.append((open_time, symbol, row))
    return results


def trade_event(
    results: Sequence[Tuple[pd.Timestamp, str, pd.Series]],
    parameter_configs: Sequence[StopParameterConfig],
) -> dict[str, list[dict]]:
    """Translate 15m signals into trades for every parameter config."""
    trades: dict[str, list[dict]] = {cfg.name: [] for cfg in parameter_configs}
    if not parameter_configs:
        return trades

    iterable = tqdm(results, desc="Simulating trades", leave=False)
    for open_time, symbol, _ in iterable:
        entry_time = open_time + pd.Timedelta(minutes=15)
        trade_results = simulate_minute_trade_multi(symbol, entry_time, parameter_configs)
        if trade_results is None:
            print(
                f"No 1m data for {symbol} starting {entry_time}, skip minute-level trade."
            )
            continue

        for config in parameter_configs:
            config_trade = trade_results.get(config.name)
            if not config_trade:
                continue
            pnl = (config_trade["exit_price"] - config_trade["entry_price"]) / config_trade[
                "entry_price"
            ]
            pnl *= -1
            pnl = pnl - (pnl + 2) * 0.0005
            config_trade = {
                **config_trade,
                "pnl": pnl,
                "symbol": symbol,
                "holding_minutes": (
                    (config_trade["exit_time"] - config_trade["entry_time"]).total_seconds() / 60.0
                    if config_trade["exit_time"] and config_trade["entry_time"]
                    else None
                ),
            }
            if config.metadata:
                config_trade["parameters"] = config.metadata
            trades[config.name].append(config_trade)

    return trades


def present_pnl(
    trades: Sequence[dict],
    *,
    trade_log_path: Path = Path("./result/test_1/trade_log.pkl"),
    curve_path: Path = Path("./result/test_1/pnl_curve.pdf"),
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> Optional[dict[str, Any]]:
    """Summarize trade performance, export logs, plot the PnL curve, and return stats."""
    if not trades:
        print("No trades generated; skip presentation.")
        return None

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holding_minutes = [t.get("holding_minutes") for t in trades if t.get("holding_minutes") is not None]

    win_rate = len(wins) / len(trades)
    avg_pnl = sum(pnls) / len(pnls)
    avg_holding = sum(holding_minutes) / len(holding_minutes) if holding_minutes else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    profit_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("nan")
    stats_text = (
        "Trade stats:\n"
        f"Count: {len(trades)}\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Avg PnL: {avg_pnl:.2%}\n"
        f"Avg Hold: {avg_holding:.1f}m\n"
        f"Avg Win: {avg_win:.2%}\n"
        f"Avg Loss: {avg_loss:.2%}\n"
        f"P/L Ratio: {profit_loss_ratio:.2f}"
    )
    print(stats_text.replace("\n", " "))
    stats_data: dict[str, Any] = {
        "count": len(trades),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_holding": avg_holding,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": profit_loss_ratio,
        "stats_text": stats_text,
    }

    trade_log_path.parent.mkdir(parents=True, exist_ok=True)
    trade_df = pd.DataFrame(trades)
    trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"])
    trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])
    trade_df = trade_df.sort_values(by="entry_time")
    trade_df.to_pickle(trade_log_path)
    print(f"Saved trade log to {trade_log_path}")

    if start_time is None or end_time is None:
        return stats_data

    start_ts = _normalize_timestamp(start_time).normalize()
    end_ts = _normalize_timestamp(end_time).normalize()
    daily_index = pd.date_range(start_ts, end_ts, freq="D")
    trade_df["entry_date"] = trade_df["entry_time"].dt.normalize()

    capital = []
    current_value = 1.0
    for day in daily_index:
        day_trades = trade_df[trade_df["entry_date"] == day]
        day_return = day_trades["pnl"].sum() * 0.1  if not day_trades.empty else 0.0
        current_value *= 1 + day_return 
        capital.append(current_value)

    capital_series = pd.Series(capital, index=daily_index)
    plot_average_pnl(
        capital_series,
        f"Average PnL {start_ts.date()} to {end_ts.date()}",
        output_path=curve_path,
        stats_text=stats_text,
    )
    print(f"Saved PnL curve to {curve_path}")
    return stats_data


def plot_average_pnl(
    pnl_series: pd.Series,
    title: str,
    *,
    output_path: Optional[Path] = None,
    stats_text: Optional[str] = None,
) -> None:
    if len(pnl_series) < 2:
        raise ValueError("Insufficient data points to plot.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    ax1.plot(pnl_series.index, pnl_series.values, linewidth=2, color="#2E86AB", label="Average PnL")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Initial")
    ax1.fill_between(
        pnl_series.index,
        1.0,
        pnl_series.values,
        where=(pnl_series.values >= 1.0),
        alpha=0.3,
        color="green",
        interpolate=True,
    )
    ax1.fill_between(
        pnl_series.index,
        1.0,
        pnl_series.values,
        where=(pnl_series.values < 1.0),
        alpha=0.3,
        color="red",
        interpolate=True,
    )

    ax1.set_title(title, fontsize=16, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("PnL (Cumulative)", fontsize=12)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    total_return = (pnl_series.iloc[-1] / pnl_series.iloc[0] - 1) * 100
    summary_text = f"Total Return: {total_return:.2f}%\nFinal PnL: {pnl_series.iloc[-1]:.4f}"
    ax1.text(
        0.02,
        0.98,
        summary_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    if stats_text is not None:
        ax1.text(
            0.98,
            0.02,
            stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

    running_max = pnl_series.expanding().max()
    drawdown = (pnl_series - running_max) / running_max * 100
    ax2.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.5)

    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    max_dd = drawdown.min()
    ax2.text(
        0.02,
        0.02,
        f"Max Drawdown: {max_dd:.2f}%",
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Example usage
    START = "2024-01-01"
    END = "2025-11-30"
    result = find_events(START, END, lambda df: condition(df))

    parameter_runs: list[dict[str, Any]] = []

    for A in [i / 1000 for i in range(66, 80, 2)]:
        for B in [i / 1000 for i in range(31)]:
            for C in [i / 10000 for i in range(5, 12, 1)]:
                name = f"paras_{A}_{B}_{C}"
                stop_fn = make_stop_level_fn(
                    take_profit_buffer=A,
                    stop_loss_buffer=B,
                    trailing_factor=C,
                )
                config = StopParameterConfig(
                    name=name,
                    determine_stop_fn=stop_fn,
                    metadata={"A": A, "B": B, "C": C},
                )
                parameter_runs.append(
                    {
                        "config": config,
                        "trade_log_path": Path(f"./result/{START}_{END}/c2/{name}/trade_log.pkl"),
                        "curve_path": Path(f"./result/{START}_{END}/c2/{name}/pnl_curve.pdf"),
                    }
                )

    trades_by_config = trade_event(result, [run["config"] for run in parameter_runs])
    summary_records: list[dict[str, Any]] = []

    for run in parameter_runs:
        config: StopParameterConfig = run["config"]
        trades = trades_by_config.get(config.name, [])
        stats_data = present_pnl(
            trades,
            trade_log_path=run["trade_log_path"],
            curve_path=run["curve_path"],
            start_time=START,
            end_time=END,
        )
        if not stats_data:
            continue
        summary_records.append(
            {
                "config": config.name,
                **(config.metadata or {}),
                **{k: v for k, v in stats_data.items()},
            }
        )
    if summary_records:
        combined_df = pd.DataFrame(summary_records)
        summary_path = Path(f"./result/{START}_{END}/c2/stats_summary.pkl")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_pickle(summary_path)
