"""
Mars账户余额可视化脚本

读取 `mars_normalized.csv`，转换为 Pandas Series，并绘制累计盈亏/回撤图后保存为 PDF。
"""

from __future__ import annotations

import os
from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = BASE_DIR / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

try:
    plt.style.use("seaborn-v0_8")
except OSError:
    plt.style.use("seaborn")

FONT_PATH = BASE_DIR / "华文仿宋.ttf"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_chinese_font() -> FontProperties | None:
    """返回可用于绘图的中文字体属性。"""
    if FONT_PATH.exists():
        try:
            prop = FontProperties(fname=str(FONT_PATH))
            prop.set_name("MarsFangSong")
            prop.set_family("MarsFangSong")
            return prop
        except Exception:
            return None
    return None


CHINESE_FONT_PROP = load_chinese_font()


def load_mars_series(csv_path: Path) -> pd.Series:
    """读取 CSV 并返回按时间排序的余额 Series。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
    return pd.Series(
        data=df["total_balance"].astype(float).values,
        index=pd.DatetimeIndex(df["timestamp"]).tz_localize(None),
        name="total_balance",
    )


def calculate_performance_metrics(pnl_series: pd.Series) -> dict[str, float]:
    """计算最终收益、平均日收益、平均年化收益、日胜率和夏普比率。"""
    base_pnl = float(pnl_series.iloc[0])
    final_pnl = float(pnl_series.iloc[-1])
    total_return = (final_pnl / base_pnl - 1.0) * 100
    total_days = (pnl_series.index[-1] - pnl_series.index[0]).total_seconds() / 86400.0
    if total_days <= 0:
        avg_annualized_return = 0.0
    else:
        avg_annualized_return = float(((final_pnl / base_pnl) ** (365.0 / total_days) - 1.0) * 100)

    daily_balance = pnl_series.resample("D").last().dropna()
    daily_returns = daily_balance.pct_change().dropna()

    if daily_returns.empty:
        avg_daily_return = 0.0
        daily_win_rate = 0.0
        sharpe_ratio = 0.0
    else:
        avg_daily_return_decimal = float(daily_returns.mean())
        avg_daily_return = float(avg_daily_return_decimal * 100)
        avg_annualized_return = float(((1.0 + avg_daily_return_decimal) ** 365 - 1.0) * 100)
        daily_win_rate = float((daily_returns > 0).mean() * 100)
        daily_volatility = float(daily_returns.std(ddof=0))
        if math.isclose(daily_volatility, 0.0, abs_tol=1e-12):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float(daily_returns.mean() / daily_volatility * math.sqrt(365))

    return {
        "total_return": total_return,
        "avg_daily_return": avg_daily_return,
        "avg_annualized_return": avg_annualized_return,
        "daily_win_rate": daily_win_rate,
        "sharpe_ratio": sharpe_ratio,
        "total_days": total_days,
    }


def plot_average_pnl(
    pnl_series: pd.Series,
    title: str,
    output_file: Path,
) -> Path:
    """绘制累计盈亏与回撤，并保存为 PDF。"""
    if len(pnl_series) < 2:
        raise ValueError("Insufficient data points to plot.")

    base_pnl = float(pnl_series.iloc[0])
    final_pnl = float(pnl_series.iloc[-1])
    max_pnl = float(pnl_series.max())
    min_pnl = float(pnl_series.min())
    metrics = calculate_performance_metrics(pnl_series)
    print(f"[INFO] 初始余额: {base_pnl:.4f}, 最终余额: {final_pnl:.4f}, 最高余额: {max_pnl:.4f}, 最低余额: {min_pnl:.4f}")
    print(
        "[INFO] 最终收益: "
        f"{metrics['total_return']:.4f}%, "
        f"平均日收益率: {metrics['avg_daily_return']:.4f}%, "
        f"平均年化收益率: {metrics['avg_annualized_return']:.4f}%, "
        f"夏普比率: {metrics['sharpe_ratio']:.4f}, "
        f"总运行天数: {metrics['total_days']:.2f} 天, "
        f"日胜率: {metrics['daily_win_rate']:.2f}%"
    )
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    font_prop = CHINESE_FONT_PROP
    font_kwargs = {"fontproperties": font_prop} if font_prop else {}

    ax1.plot(
        pnl_series.index,
        pnl_series.values,
        linewidth=0.1,
        color="#2E86AB",
        label="余额变化",
    )
    ax1.axhline(
        y=base_pnl,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="初始余额",
    )
    ax1.fill_between(
        pnl_series.index,
        base_pnl,
        pnl_series.values,
        where=pnl_series.values >= base_pnl,
        alpha=0.3,
        color="green",
        interpolate=True,
    )
    ax1.fill_between(
        pnl_series.index,
        base_pnl,
        pnl_series.values,
        where=pnl_series.values < base_pnl,
        alpha=0.3,
        color="red",
        interpolate=True,
    )

    legend_kwargs = {"prop": font_prop} if font_prop else {}
    ax1.set_title(title, fontsize=20, fontweight="bold", **font_kwargs)
    ax1.set_xlabel("日期", fontsize=12, **font_kwargs)
    ax1.set_ylabel("余额", fontsize=12, **font_kwargs)
    ax1.legend(loc="best", fontsize=10, **legend_kwargs)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    tick_labels_ax1 = ax1.xaxis.get_majorticklabels()
    plt.setp(tick_labels_ax1, rotation=45, ha="right")
    if font_prop:
        for label in tick_labels_ax1 + ax1.yaxis.get_majorticklabels():
            label.set_fontproperties(font_prop)

    stats_text = (
        f"当前收益率： {metrics['total_return']:.4f}%\n"
        f"平均日收益率： {metrics['avg_daily_return']:.3f}%\n"
        f"平均年化收益率： {metrics['avg_annualized_return']:.1f}%\n"
        f"夏普比率： {metrics['sharpe_ratio']:.4f}\n"
        f"总运行天数： {metrics['total_days']:.2f} 天\n"
        f"日胜率： {metrics['daily_win_rate']:.2f}%"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        **font_kwargs,
    )

    running_max = pnl_series.expanding().max()
    drawdown = (pnl_series - running_max) / running_max * 100
    ax2.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.5)

    ax2.set_xlabel("日期", fontsize=12, **font_kwargs)
    ax2.set_ylabel("回撤 (%)", fontsize=12, **font_kwargs)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    tick_labels_ax2 = ax2.xaxis.get_majorticklabels()
    plt.setp(tick_labels_ax2, rotation=45, ha="right")
    if font_prop:
        for label in tick_labels_ax2 + ax2.yaxis.get_majorticklabels():
            label.set_fontproperties(font_prop)

    max_dd = drawdown.min()
    ax2.text(
        0.02,
        0.05,
        f"最大回撤： {max_dd:.2f}%",
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        **font_kwargs,
    )

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="pdf", bbox_inches="tight", dpi=5000)
    plt.close(fig)
    return output_file


CSV_PATH = BASE_DIR / "mars_normalized.csv"
OUTPUT_PDF_PATH = BASE_DIR / "mars_monitor.pdf"
CHART_TITLE = "Mars 账户余额变化"


def main() -> None:
    pnl_series = load_mars_series(CSV_PATH)
    saved_path = plot_average_pnl(pnl_series, CHART_TITLE, OUTPUT_PDF_PATH)
    print(f"[OK] 图表已保存: {saved_path}")


if __name__ == "__main__":
    main()
