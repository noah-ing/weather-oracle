"""Backtesting framework for Weather Oracle edge detection."""

from src.backtesting.backtest import (
    BacktestResult,
    SimulatedMarket,
    run_backtest,
    generate_backtest_report,
)

__all__ = [
    "BacktestResult",
    "SimulatedMarket",
    "run_backtest",
    "generate_backtest_report",
]
