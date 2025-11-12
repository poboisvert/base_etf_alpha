from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from portwine.analyzers import Analyzer


@dataclass
class TrendFilterResult:
    normalized_prices: pd.DataFrame
    moving_average: pd.DataFrame
    filter_mask: pd.DataFrame


class TrendFilterAnalyzer(Analyzer):
    """
    Simple trend filter analyzer that compares a normalized price series to a
    moving average and highlights periods where the filter condition is met.

    The analyzer expects the standard backtester results dictionary containing
    ``tickers_returns`` (daily percentage returns). Prices are reconstructed
    from the cumulative product of ``1 + returns`` so that the filter can be
    applied consistently even if the backtest worked purely with returns.
    """

    def __init__(self, filter_type: str = "price_above_ma", lookback: int = 200, max_plot_tickers: int = 6):
        self.filter_type = filter_type
        self.lookback = lookback
        self.max_plot_tickers = max_plot_tickers

    def analyze(self, results: Dict) -> TrendFilterResult | None:
        tickers_returns: pd.DataFrame | None = results.get("tickers_returns")
        if tickers_returns is None or tickers_returns.empty:
            print("TrendFilterAnalyzer: 'tickers_returns' missing or empty in results.")
            return None

        # Reconstruct a normalized price series (starts at 1.0)
        norm_prices = (1.0 + tickers_returns.fillna(0.0)).cumprod()

        if self.filter_type != "price_above_ma":
            print(f"TrendFilterAnalyzer: Unsupported filter_type '{self.filter_type}'. Using 'price_above_ma' instead.")
            self.filter_type = "price_above_ma"

        moving_average = norm_prices.rolling(self.lookback, min_periods=1).mean()
        filter_mask = norm_prices > moving_average

        return TrendFilterResult(
            normalized_prices=norm_prices,
            moving_average=moving_average,
            filter_mask=filter_mask,
        )

    def plot(self, results: Dict) -> None:
        analysis = self.analyze(results)
        if analysis is None:
            return

        norm_prices = analysis.normalized_prices
        moving_average = analysis.moving_average
        filter_mask = analysis.filter_mask

        tickers = list(norm_prices.columns)
        if not tickers:
            print("TrendFilterAnalyzer: No tickers available to plot.")
            return

        n_plots = min(len(tickers), self.max_plot_tickers)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for ax, ticker in zip(axes, tickers[:n_plots]):
            ax.plot(norm_prices.index, norm_prices[ticker], label=f"{ticker} (normalized price)", color="tab:blue")
            ax.plot(moving_average.index, moving_average[ticker], label=f"{self.lookback}-day MA", color="tab:orange", linewidth=1.5)

            # Highlight where the filter condition is true
            ax.fill_between(
                norm_prices.index,
                norm_prices[ticker],
                moving_average[ticker],
                where=filter_mask[ticker],
                color="tab:green",
                alpha=0.15,
                label="Price above MA",
            )

            ax.set_title(f"{ticker} Trend Filter ({self.filter_type})")
            ax.set_ylabel("Normalized Price")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        fig.tight_layout()
        plt.show()

