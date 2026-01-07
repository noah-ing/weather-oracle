"""Backtesting framework for Weather Oracle edge detection.

This module simulates historical Kalshi markets using past weather data
to test if our edge detection strategy would have been profitable.

Approach:
1. Generate simulated markets from historical observations
   - Create temp_high/temp_low markets for each day/city
   - Use realistic thresholds based on climatology
2. For each simulated market:
   - Get the day's actual high/low from observations
   - Calculate what our model would have predicted
   - Determine if our edge detection would have bet correctly
3. Calculate overall performance metrics:
   - Win rate
   - Average edge captured
   - Simulated P/L
   - Best/worst conditions

Usage:
    from src.backtesting.backtest import run_backtest, generate_backtest_report

    results = run_backtest(days=30)
    report = generate_backtest_report(results)
    print(report)

CLI:
    python -m src.cli backtest --days 30
"""

import sqlite3
import statistics
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from src.config import DB_PATH
from src.db.database import get_connection
from src.calibration.probability import get_threshold_probability, CalibrationConfig


@dataclass
class SimulatedMarket:
    """A simulated Kalshi-style weather market for backtesting.

    Attributes:
        location: City name (e.g., "New York, NY")
        target_date: Date the market is about
        condition_type: "temp_high" or "temp_low"
        threshold: Temperature threshold (F)
        comparison: ">" or "<"
        actual_temp: What the actual temperature was
        outcome: True if condition was met (YES wins), False otherwise
    """
    location: str
    target_date: date
    condition_type: str
    threshold: float
    comparison: str
    actual_temp: float
    outcome: bool  # True = YES wins, False = NO wins


@dataclass
class BacktestTrade:
    """A simulated trade in the backtest.

    Attributes:
        market: The simulated market
        predicted_temp: What our model predicted
        model_prob: Our calibrated probability
        market_prob: Simulated market probability (randomized around actual)
        edge_pct: Our calculated edge
        side: "YES" or "NO" - which side we bet
        kelly_fraction: Kelly-optimal bet size
        won: Whether the trade was profitable
        pnl: Profit/loss (normalized to $1 stake)
    """
    market: SimulatedMarket
    predicted_temp: float
    model_prob: float
    market_prob: float
    edge_pct: float
    side: str
    kelly_fraction: float
    won: bool
    pnl: float


@dataclass
class ConditionPerformance:
    """Performance metrics for a specific condition type.

    Attributes:
        condition_type: "temp_high", "temp_low", etc.
        trades: Number of trades
        wins: Number of winning trades
        win_rate: Fraction of trades that won
        avg_edge: Average edge on trades
        total_pnl: Total P/L
        avg_pnl: Average P/L per trade
    """
    condition_type: str
    trades: int
    wins: int
    win_rate: float
    avg_edge: float
    total_pnl: float
    avg_pnl: float


@dataclass
class LocationPerformance:
    """Performance metrics for a specific location.

    Attributes:
        location: City name
        trades: Number of trades
        wins: Number of winning trades
        win_rate: Fraction of trades that won
        avg_edge: Average edge on trades
        total_pnl: Total P/L
    """
    location: str
    trades: int
    wins: int
    win_rate: float
    avg_edge: float
    total_pnl: float


@dataclass
class BacktestResult:
    """Complete backtest results.

    Attributes:
        start_date: First date in backtest
        end_date: Last date in backtest
        total_days: Number of days simulated
        total_markets: Number of simulated markets
        total_trades: Number of trades taken (where edge > threshold)
        wins: Number of winning trades
        losses: Number of losing trades
        win_rate: Overall win rate
        avg_edge: Average edge on trades
        avg_pnl: Average P/L per trade
        total_pnl: Total P/L (normalized to $1 stakes)
        sharpe_ratio: Risk-adjusted return (if enough trades)
        confidence_interval: 95% CI on expected profit per trade
        by_condition: Performance breakdown by condition type
        by_location: Performance breakdown by location
        trades: All individual trades
        best_conditions: Conditions where model performs best
        worst_conditions: Conditions where model performs worst
    """
    start_date: date
    end_date: date
    total_days: int
    total_markets: int
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_edge: float
    avg_pnl: float
    total_pnl: float
    sharpe_ratio: Optional[float]
    confidence_interval: Tuple[float, float]
    by_condition: List[ConditionPerformance] = field(default_factory=list)
    by_location: List[LocationPerformance] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    best_conditions: List[str] = field(default_factory=list)
    worst_conditions: List[str] = field(default_factory=list)


def _get_historical_observations(
    start_date: date,
    end_date: date,
) -> Dict[str, Dict[date, Dict[str, float]]]:
    """Get historical observations grouped by location and date.

    Returns:
        Dict of {location: {date: {"high": temp, "low": temp}}}
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT city, DATE(timestamp) as obs_date, MAX(temp) as high_temp, MIN(temp) as low_temp
        FROM observations
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY city, DATE(timestamp)
        ORDER BY city, obs_date
    """
    cursor.execute(query, (start_date.isoformat(), end_date.isoformat()))
    rows = cursor.fetchall()
    conn.close()

    observations: Dict[str, Dict[date, Dict[str, float]]] = defaultdict(dict)

    for row in rows:
        city = row["city"]
        obs_date = date.fromisoformat(row["obs_date"])
        # Convert from Celsius to Fahrenheit
        high_c = row["high_temp"]
        low_c = row["low_temp"]
        high_f = high_c * 9/5 + 32
        low_f = low_c * 9/5 + 32
        observations[city][obs_date] = {"high": high_f, "low": low_f}

    return observations


def _generate_simulated_markets(
    observations: Dict[str, Dict[date, Dict[str, float]]],
    thresholds_per_day: int = 3,
) -> List[SimulatedMarket]:
    """Generate simulated markets from historical observations.

    For each day and location, creates multiple markets with different thresholds.
    Uses realistic threshold choices based on the actual temperature.

    Args:
        observations: Historical temp data by location and date
        thresholds_per_day: Number of threshold markets per condition per day

    Returns:
        List of SimulatedMarket objects
    """
    import random
    random.seed(42)  # Reproducible

    markets = []

    for location, dates in observations.items():
        for obs_date, temps in dates.items():
            actual_high = temps["high"]
            actual_low = temps["low"]

            # Generate temp_high markets with thresholds near actual
            for i in range(thresholds_per_day):
                # Thresholds: below, at, and above actual
                offset = (i - 1) * 5  # -5, 0, +5
                threshold = round(actual_high + offset)

                outcome = actual_high > threshold

                markets.append(SimulatedMarket(
                    location=location,
                    target_date=obs_date,
                    condition_type="temp_high",
                    threshold=threshold,
                    comparison=">",
                    actual_temp=actual_high,
                    outcome=outcome,
                ))

            # Generate temp_low markets
            for i in range(thresholds_per_day):
                offset = (i - 1) * 5  # -5, 0, +5
                threshold = round(actual_low + offset)

                # For low markets, outcome is True if actual < threshold
                outcome = actual_low < threshold

                markets.append(SimulatedMarket(
                    location=location,
                    target_date=obs_date,
                    condition_type="temp_low",
                    threshold=threshold,
                    comparison="<",
                    actual_temp=actual_low,
                    outcome=outcome,
                ))

    return markets


def _simulate_model_prediction(
    market: SimulatedMarket,
    noise_std: float = 3.0,
) -> float:
    """Simulate what our model would have predicted.

    Adds realistic noise to the actual temperature to simulate
    what a forecast made 1-2 days ahead would have predicted.

    Args:
        market: The simulated market
        noise_std: Standard deviation of forecast error (F)

    Returns:
        Simulated model prediction temperature
    """
    import random

    # Add Gaussian noise to simulate forecast error
    # This represents our model's typical error
    noise = random.gauss(0, noise_std)
    predicted = market.actual_temp + noise

    return round(predicted, 1)


def _simulate_market_probability(
    market: SimulatedMarket,
    efficiency: float = 0.8,
) -> float:
    """Simulate a Kalshi market probability.

    Creates a market probability that's somewhat accurate but not perfectly
    efficient, representing real market conditions.

    Args:
        market: The simulated market
        efficiency: How accurate the market is (1.0 = perfectly efficient)

    Returns:
        Simulated market probability (0.05-0.95)
    """
    import random

    # Calculate "true" probability based on actual outcome
    # In reality, the market doesn't know the outcome, but it's somewhat correlated
    true_prob = 1.0 if market.outcome else 0.0

    # Market is not perfectly efficient - mix true prob with noise
    noise_prob = random.uniform(0.3, 0.7)
    market_prob = efficiency * true_prob + (1 - efficiency) * noise_prob

    # Add some random noise
    market_prob += random.gauss(0, 0.1)

    # Clamp to valid range
    return max(0.05, min(0.95, market_prob))


def _calculate_trade(
    market: SimulatedMarket,
    predicted_temp: float,
    market_prob: float,
    min_edge: float = 10.0,
) -> Optional[BacktestTrade]:
    """Calculate whether we would trade this market and the result.

    Args:
        market: The simulated market
        predicted_temp: Our model's predicted temperature
        market_prob: Simulated market probability
        min_edge: Minimum edge to take a trade

    Returns:
        BacktestTrade if we would trade, None otherwise
    """
    # Calculate our calibrated probability
    temp_type = "high" if market.condition_type == "temp_high" else "low"

    model_prob = get_threshold_probability(
        predicted_temp=predicted_temp,
        threshold=market.threshold,
        comparison=market.comparison,
        temp_type=temp_type,
        lead_days=1,
    )

    # Calculate edge
    edge_pct = (model_prob - market_prob) / max(market_prob, 0.01) * 100

    # Skip if edge is too small
    if abs(edge_pct) < min_edge:
        return None

    # Determine side
    side = "YES" if model_prob > market_prob else "NO"

    # Calculate Kelly fraction
    if side == "YES":
        kelly = (model_prob - market_prob) / max(1 - market_prob, 0.01)
    else:
        kelly = (market_prob - model_prob) / max(market_prob, 0.01)

    kelly = max(0.0, min(0.25, kelly))  # Cap at 25%

    # Determine if trade won
    if side == "YES":
        won = market.outcome
        if won:
            pnl = (1 - market_prob) / market_prob  # Win payout
        else:
            pnl = -1.0  # Lose stake
    else:
        won = not market.outcome
        if won:
            pnl = market_prob / (1 - market_prob)  # Win payout
        else:
            pnl = -1.0  # Lose stake

    return BacktestTrade(
        market=market,
        predicted_temp=predicted_temp,
        model_prob=model_prob,
        market_prob=market_prob,
        edge_pct=edge_pct,
        side=side,
        kelly_fraction=kelly,
        won=won,
        pnl=pnl,
    )


def _calculate_confidence_interval(
    pnls: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate confidence interval on mean P/L.

    Uses t-distribution for small samples.

    Args:
        pnls: List of P/L values
        confidence: Confidence level

    Returns:
        Tuple of (lower, upper) bounds on mean
    """
    if len(pnls) < 2:
        return (0.0, 0.0)

    n = len(pnls)
    mean = statistics.mean(pnls)
    std = statistics.stdev(pnls)
    se = std / math.sqrt(n)

    # Use normal approximation for z-score
    # For 95% CI, z = 1.96
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    lower = mean - z * se
    upper = mean + z * se

    return (lower, upper)


def _analyze_by_condition(trades: List[BacktestTrade]) -> List[ConditionPerformance]:
    """Analyze performance by condition type."""
    by_condition: Dict[str, List[BacktestTrade]] = defaultdict(list)

    for trade in trades:
        by_condition[trade.market.condition_type].append(trade)

    results = []
    for cond_type, cond_trades in by_condition.items():
        wins = sum(1 for t in cond_trades if t.won)
        pnls = [t.pnl for t in cond_trades]
        edges = [t.edge_pct for t in cond_trades]

        results.append(ConditionPerformance(
            condition_type=cond_type,
            trades=len(cond_trades),
            wins=wins,
            win_rate=wins / len(cond_trades) if cond_trades else 0,
            avg_edge=statistics.mean(edges) if edges else 0,
            total_pnl=sum(pnls),
            avg_pnl=statistics.mean(pnls) if pnls else 0,
        ))

    return sorted(results, key=lambda x: x.total_pnl, reverse=True)


def _analyze_by_location(trades: List[BacktestTrade]) -> List[LocationPerformance]:
    """Analyze performance by location."""
    by_location: Dict[str, List[BacktestTrade]] = defaultdict(list)

    for trade in trades:
        by_location[trade.market.location].append(trade)

    results = []
    for location, loc_trades in by_location.items():
        wins = sum(1 for t in loc_trades if t.won)
        pnls = [t.pnl for t in loc_trades]
        edges = [t.edge_pct for t in loc_trades]

        results.append(LocationPerformance(
            location=location,
            trades=len(loc_trades),
            wins=wins,
            win_rate=wins / len(loc_trades) if loc_trades else 0,
            avg_edge=statistics.mean(edges) if edges else 0,
            total_pnl=sum(pnls),
        ))

    return sorted(results, key=lambda x: x.total_pnl, reverse=True)


def run_backtest(
    days: int = 30,
    min_edge: float = 10.0,
    thresholds_per_day: int = 3,
    forecast_noise_std: float = 3.0,
    market_efficiency: float = 0.8,
    verbose: bool = True,
) -> BacktestResult:
    """Run a complete backtest simulation.

    Simulates Kalshi markets using historical weather data and tests
    if our edge detection strategy would have been profitable.

    Args:
        days: Number of days to backtest
        min_edge: Minimum edge percentage to take a trade
        thresholds_per_day: Number of threshold variations per condition per day
        forecast_noise_std: Standard deviation of simulated forecast error
        market_efficiency: How accurate the simulated market is (0-1)
        verbose: Whether to print progress

    Returns:
        BacktestResult with complete analysis

    Example:
        >>> results = run_backtest(days=30)
        >>> print(f"Win rate: {results.win_rate:.1%}")
        >>> print(f"Total P/L: ${results.total_pnl:.2f}")
    """
    if verbose:
        print(f"Running backtest for {days} days...")

    # Determine date range
    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=days)

    if verbose:
        print(f"  Date range: {start_date} to {end_date}")

    # Get historical observations
    if verbose:
        print("  Loading historical observations...")

    observations = _get_historical_observations(start_date, end_date)

    if not observations:
        print("  No observations found in date range!")
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_days=0,
            total_markets=0,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            avg_edge=0.0,
            avg_pnl=0.0,
            total_pnl=0.0,
            sharpe_ratio=None,
            confidence_interval=(0.0, 0.0),
        )

    # Count unique dates across all locations
    all_dates = set()
    for loc_dates in observations.values():
        all_dates.update(loc_dates.keys())
    total_days = len(all_dates)

    if verbose:
        print(f"  Found {len(observations)} locations, {total_days} days of data")

    # Generate simulated markets
    if verbose:
        print("  Generating simulated markets...")

    markets = _generate_simulated_markets(observations, thresholds_per_day)

    if verbose:
        print(f"  Generated {len(markets)} simulated markets")

    # Run trades
    if verbose:
        print("  Running edge detection on simulated markets...")

    import random
    random.seed(42)  # Reproducible

    trades: List[BacktestTrade] = []

    for i, market in enumerate(markets):
        if verbose and i > 0 and i % 1000 == 0:
            print(f"    Processed {i}/{len(markets)} markets...")

        # Simulate our model's prediction
        predicted_temp = _simulate_model_prediction(market, forecast_noise_std)

        # Simulate market probability
        market_prob = _simulate_market_probability(market, market_efficiency)

        # Calculate if we would trade
        trade = _calculate_trade(market, predicted_temp, market_prob, min_edge)

        if trade:
            trades.append(trade)

    if verbose:
        print(f"  Took {len(trades)} trades (edges >= {min_edge}%)")

    # Calculate overall statistics
    if trades:
        wins = sum(1 for t in trades if t.won)
        losses = len(trades) - wins
        win_rate = wins / len(trades)
        avg_edge = statistics.mean([t.edge_pct for t in trades])
        pnls = [t.pnl for t in trades]
        avg_pnl = statistics.mean(pnls)
        total_pnl = sum(pnls)

        # Sharpe ratio (if enough trades)
        if len(pnls) > 10:
            sharpe = statistics.mean(pnls) / statistics.stdev(pnls) if statistics.stdev(pnls) > 0 else 0
            sharpe_ratio = sharpe * math.sqrt(252)  # Annualized
        else:
            sharpe_ratio = None

        confidence_interval = _calculate_confidence_interval(pnls)
    else:
        wins = 0
        losses = 0
        win_rate = 0.0
        avg_edge = 0.0
        avg_pnl = 0.0
        total_pnl = 0.0
        sharpe_ratio = None
        confidence_interval = (0.0, 0.0)

    # Analyze by condition and location
    by_condition = _analyze_by_condition(trades)
    by_location = _analyze_by_location(trades)

    # Identify best/worst conditions
    best_conditions = [c.condition_type for c in by_condition[:2] if c.avg_pnl > 0]
    worst_conditions = [c.condition_type for c in by_condition[-2:] if c.avg_pnl < 0]

    return BacktestResult(
        start_date=start_date,
        end_date=end_date,
        total_days=total_days,
        total_markets=len(markets),
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_edge=avg_edge,
        avg_pnl=avg_pnl,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        confidence_interval=confidence_interval,
        by_condition=by_condition,
        by_location=by_location,
        trades=trades,
        best_conditions=best_conditions,
        worst_conditions=worst_conditions,
    )


def generate_backtest_report(result: BacktestResult) -> str:
    """Generate a formatted backtest report.

    Args:
        result: BacktestResult from run_backtest

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "BACKTEST REPORT - Weather Oracle Edge Detection",
        "=" * 70,
        "",
        f"Period: {result.start_date} to {result.end_date} ({result.total_days} days)",
        f"Simulated Markets: {result.total_markets:,}",
        f"Trades Taken: {result.total_trades:,}",
        "",
        "OVERALL PERFORMANCE",
        "-" * 70,
        f"Win Rate:      {result.win_rate:.1%} ({result.wins} wins, {result.losses} losses)",
        f"Average Edge:  {result.avg_edge:.1f}%",
        f"Average P/L:   ${result.avg_pnl:.3f} per $1 bet",
        f"Total P/L:     ${result.total_pnl:.2f} (sum of all trades)",
        "",
    ]

    if result.sharpe_ratio is not None:
        lines.append(f"Sharpe Ratio:  {result.sharpe_ratio:.2f} (annualized)")

    lines.extend([
        "",
        f"95% CI on Expected Profit: ${result.confidence_interval[0]:.3f} to ${result.confidence_interval[1]:.3f} per trade",
        "",
    ])

    # Profitability assessment
    if result.confidence_interval[0] > 0:
        lines.append("[PROFITABLE] Lower bound of CI is positive - strategy likely profitable")
    elif result.confidence_interval[1] < 0:
        lines.append("[UNPROFITABLE] Upper bound of CI is negative - strategy likely losing money")
    else:
        lines.append("[UNCERTAIN] CI spans zero - more data needed to confirm profitability")

    lines.extend([
        "",
        "PERFORMANCE BY CONDITION TYPE",
        "-" * 70,
        f"{'Condition':<12} {'Trades':>8} {'Win%':>8} {'Avg Edge':>10} {'Total P/L':>12}",
    ])

    for cond in result.by_condition:
        lines.append(
            f"{cond.condition_type:<12} "
            f"{cond.trades:>8} "
            f"{cond.win_rate:>7.1%} "
            f"{cond.avg_edge:>+9.1f}% "
            f"${cond.total_pnl:>+11.2f}"
        )

    lines.extend([
        "",
        "PERFORMANCE BY LOCATION (Top 5 / Bottom 5)",
        "-" * 70,
        f"{'Location':<25} {'Trades':>8} {'Win%':>8} {'Total P/L':>12}",
    ])

    # Top 5
    for loc in result.by_location[:5]:
        lines.append(
            f"{loc.location:<25} "
            f"{loc.trades:>8} "
            f"{loc.win_rate:>7.1%} "
            f"${loc.total_pnl:>+11.2f}"
        )

    if len(result.by_location) > 10:
        lines.append("...")

    # Bottom 5
    for loc in result.by_location[-5:]:
        lines.append(
            f"{loc.location:<25} "
            f"{loc.trades:>8} "
            f"{loc.win_rate:>7.1%} "
            f"${loc.total_pnl:>+11.2f}"
        )

    lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 70,
    ])

    if result.best_conditions:
        lines.append(f"Best conditions: {', '.join(result.best_conditions)}")
    if result.worst_conditions:
        lines.append(f"Avoid conditions: {', '.join(result.worst_conditions)}")

    # Trading recommendations
    if result.win_rate > 0.55 and result.avg_pnl > 0:
        lines.append("Strategy appears profitable - consider live trading with small stakes")
    elif result.win_rate > 0.50:
        lines.append("Strategy shows promise but needs refinement - increase min_edge threshold")
    else:
        lines.append("Strategy underperforming - review model calibration and edge calculation")

    lines.extend([
        "",
        "=" * 70,
        "Note: Backtest uses simulated markets. Real performance may differ.",
        "=" * 70,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Weather Oracle Backtest[/bold cyan]\n"
        "Testing edge detection on historical data",
        border_style="cyan"
    ))

    # Run backtest
    result = run_backtest(days=30, min_edge=10.0, verbose=True)

    # Print report
    report = generate_backtest_report(result)
    console.print("\n")
    console.print(Panel(report, title="Backtest Results", border_style="green"))
