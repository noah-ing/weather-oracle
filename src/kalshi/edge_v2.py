"""Upgraded edge calculator with calibrated probabilities (V2).

This module replaces the naive sigmoid-based probability calculation with
proper probabilistic calibration based on historical forecast errors.

Key improvements over V1:
- Uses ensemble_v2 for multi-model predictions instead of single model
- Uses probabilistic calibration for threshold probability (Gaussian CDF)
- Implements Kelly criterion for optimal bet sizing
- Improved confidence scoring based on model agreement, historical accuracy, volume
- Only surfaces edges where calibrated probability differs significantly from market

Usage:
    from src.kalshi.edge_v2 import find_calibrated_edges

    edges = find_calibrated_edges()
    for edge in edges:
        print(f"{edge.market.ticker}: {edge.edge_pct:+.1f}% edge, Kelly: {edge.kelly_fraction:.1%}")
"""

import datetime
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from scipy.stats import norm

from src.kalshi.scanner import WeatherMarket, scan_weather_markets
from src.inference.ensemble_v2 import predict_ensemble, predict_ensemble_fast, EnsemblePrediction
from src.calibration.probability import (
    get_threshold_probability,
    get_confidence_interval,
    calibrate_market_probability,
)


# Aggressive strategy configuration
AGGRESSIVE_CONFIG = {
    "min_edge": 5.0,           # Minimum edge % to consider
    "max_yes_price": 0.85,     # Don't buy YES above 85¢ (limited upside)
    "min_yes_price": 0.15,     # Don't buy NO below 15¢ (same as YES > 85¢)
    "min_market_price": 0.15,  # Only alert on markets priced >= 15% (avoid extreme odds)
    "max_market_price": 0.85,  # Only alert on markets priced <= 85% (avoid extreme odds)
    "min_kelly": 0.02,         # 2% minimum position size
    "max_kelly": 0.25,         # 25% maximum position size
    "min_std": 2.0,            # Minimum temperature std dev (avoid overconfidence)
}


@dataclass
class CalibratedEdge:
    """Represents a calibrated edge opportunity where model disagrees with market.

    Attributes:
        market: The WeatherMarket this opportunity is for
        kalshi_prob: Market's implied probability (0-1)
        model_prob: Calibrated model probability (0-1) from ensemble + probability calibration
        edge_pct: Edge percentage: (model_prob - kalshi_prob) / kalshi_prob * 100
        expected_value: Expected value of a $1 bet
        kelly_fraction: Kelly criterion optimal bet fraction
        confidence: Confidence level (0-1) based on multiple factors
        confidence_level: Categorical: "HIGH", "MEDIUM", "LOW"
        side: "YES" if we think probability is higher than market, "NO" otherwise
        explanation: Text explaining the edge

        # Additional V2 fields
        ensemble_prediction: Full ensemble prediction details
        model_agreement: How much models agree (0-1, higher = more agreement)
        sources_used: List of sources used in ensemble
        predicted_temp: Ensemble predicted temperature for threshold comparison
        temp_std: Standard deviation of temperature predictions across models
        per_source_probs: Probability estimate from each source
    """
    market: WeatherMarket
    kalshi_prob: float
    model_prob: float
    edge_pct: float
    expected_value: float
    kelly_fraction: float
    confidence: float
    confidence_level: str
    side: str
    explanation: str

    # V2 fields
    ensemble_prediction: Optional[EnsemblePrediction] = None
    model_agreement: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    predicted_temp: Optional[float] = None
    temp_std: Optional[float] = None
    per_source_probs: Dict[str, float] = field(default_factory=dict)


def _calculate_kelly_fraction(
    model_prob: float,
    kalshi_prob: float,
    side: str,
) -> float:
    """Calculate Kelly criterion optimal bet fraction.

    Kelly criterion gives the optimal fraction of bankroll to bet:
    f* = (bp - q) / b

    Where:
    - b = net odds received on the bet (payout / stake - 1)
    - p = probability of winning (model probability)
    - q = probability of losing (1 - p)

    For binary markets:
    - YES bet at price P: payout = 1/P - 1, so f* = p - (1-p)/(1/P - 1)
    - Simplified: f* = (p - P) / (1 - P) for YES bets
    - For NO bets: f* = ((1-p) - (1-P)) / P = (P - p) / P

    Args:
        model_prob: Our probability estimate
        kalshi_prob: Market probability (price)
        side: "YES" or "NO"

    Returns:
        Kelly fraction (0 to 1), clamped to reasonable bounds
    """
    if side == "YES":
        # Betting YES at price kalshi_prob
        # Win probability: model_prob
        # Win payout: (1 - kalshi_prob) / kalshi_prob  (net odds)
        if kalshi_prob >= 0.99:  # Avoid division issues
            return 0.0
        kelly = (model_prob - kalshi_prob) / (1 - kalshi_prob)
    else:
        # Betting NO at price (1 - kalshi_prob)
        # Win probability: (1 - model_prob)
        # Win payout: kalshi_prob / (1 - kalshi_prob)  (net odds)
        if kalshi_prob <= 0.01:  # Avoid division issues
            return 0.0
        kelly = (kalshi_prob - model_prob) / kalshi_prob

    # Clamp to reasonable bounds
    # Never bet more than 25% of bankroll (fractional Kelly)
    # Negative Kelly means don't bet (edge is wrong direction)
    kelly = max(0.0, min(0.25, kelly))

    return kelly


def _calculate_confidence(
    market: WeatherMarket,
    model_prob: float,
    edge_pct: float,
    model_agreement: float,
    ensemble_prediction: Optional[EnsemblePrediction],
) -> tuple[float, str]:
    """Calculate confidence level for a calibrated edge.

    Confidence is based on multiple factors:
    1. Model agreement: Do all sources agree? (higher = more confident)
    2. Historical accuracy: Sources with good track record (from ensemble weights)
    3. Market liquidity: Volume and bid-ask spread
    4. Edge size: Larger edges in efficient markets = less confident
    5. Model probability extremity: Extreme probs (near 0 or 1) = less reliable

    Args:
        market: The weather market
        model_prob: Calibrated probability
        edge_pct: Calculated edge percentage
        model_agreement: Agreement score from ensemble (0-1)
        ensemble_prediction: Full ensemble prediction for additional context

    Returns:
        Tuple of (confidence_score 0-1, confidence_level "HIGH"/"MEDIUM"/"LOW")
    """
    confidence_score = 0.0

    # 1. Model agreement (0-35 points)
    # From ensemble_v2: confidence = 0.8-1.0 for std < 2F, 0.5-0.8 for std 2-5F
    if ensemble_prediction:
        confidence_score += model_agreement * 35
    else:
        confidence_score += 15  # Default if no ensemble

    # 2. Bid-ask spread (0-20 points)
    # Tighter spread = more efficient market = higher confidence in our signal
    spread = abs(market.yes_ask - market.yes_bid)
    if spread <= 0.02:
        confidence_score += 20
    elif spread <= 0.05:
        confidence_score += 15
    elif spread <= 0.10:
        confidence_score += 10
    else:
        confidence_score += 5

    # 3. Volume indicator (0-15 points)
    # Some volume = market is active, but very high volume = efficient
    if 100 <= market.volume <= 1000:
        confidence_score += 15  # Sweet spot
    elif 50 <= market.volume < 100:
        confidence_score += 12
    elif market.volume > 1000:
        confidence_score += 10  # High volume = more efficient
    elif 10 <= market.volume < 50:
        confidence_score += 8
    else:
        confidence_score += 3  # Very low volume = unreliable

    # 4. Model probability reasonableness (0-15 points)
    # Extreme probabilities are harder to calibrate accurately
    if 0.20 <= model_prob <= 0.80:
        confidence_score += 15
    elif 0.10 <= model_prob <= 0.90:
        confidence_score += 10
    else:
        confidence_score += 5

    # 5. Number of sources (0-15 points)
    if ensemble_prediction:
        n_sources = ensemble_prediction.total_sources
        if n_sources >= 5:
            confidence_score += 15
        elif n_sources >= 4:
            confidence_score += 12
        elif n_sources >= 3:
            confidence_score += 9
        else:
            confidence_score += 5
    else:
        confidence_score += 5

    # Normalize to 0-1
    confidence = confidence_score / 100.0
    confidence = max(0.1, min(1.0, confidence))

    # Convert to categorical
    if confidence >= 0.70:
        level = "HIGH"
    elif confidence >= 0.45:
        level = "MEDIUM"
    else:
        level = "LOW"

    return confidence, level


def _get_calibrated_probability(
    ensemble: EnsemblePrediction,
    market: WeatherMarket,
) -> tuple[float, float, Dict[str, float]]:
    """Get calibrated probability for a weather market (bracket or threshold).

    Uses the ensemble prediction with Gaussian distribution to calculate
    the probability of the weather event occurring.

    For BRACKET markets (e.g., temp 47-48°F):
        P = norm.cdf(upper, loc=temp, scale=std) - norm.cdf(lower, loc=temp, scale=std)

    For THRESHOLD markets (e.g., temp > 48°F):
        P(above) = 1 - norm.cdf(threshold, loc=temp, scale=std)
        P(below) = norm.cdf(threshold, loc=temp, scale=std)

    Args:
        ensemble: Ensemble prediction with temp forecasts
        market: Weather market with threshold/bracket info

    Returns:
        Tuple of (calibrated_prob, predicted_temp, per_source_probs)
    """
    # Determine which temperature to use based on condition type
    if market.condition_type == "temp_high":
        predicted_temp = ensemble.high_temp
        temp_std = ensemble.high_std
    elif market.condition_type == "temp_low":
        predicted_temp = ensemble.low_temp
        temp_std = ensemble.low_std
    else:
        # For non-temp markets (rain, snow), use precip probability directly
        if market.condition_type in ("rain", "snow"):
            precip_prob = ensemble.precip_probability / 100.0
            # Adjust based on threshold if available
            if market.threshold and market.threshold > 0.1:
                # Higher threshold = less likely
                precip_prob = precip_prob * 0.8
            return max(0.05, min(0.95, precip_prob)), None, {}

        return 0.5, None, {}  # Unknown condition type

    # Ensure minimum std dev to avoid overconfidence
    temp_std = max(temp_std, AGGRESSIVE_CONFIG["min_std"])

    # Add lead-time uncertainty: forecast gets worse further out
    if market.target_date:
        lead_days = (market.target_date - datetime.date.today()).days
        lead_days = max(1, lead_days)
        # Add 0.5°F per day of lead time
        temp_std += 0.5 * (lead_days - 1)

    # Calculate probability based on market type
    if market.market_type == "bracket":
        # BRACKET market: P(lower <= temp < upper)
        # Use Gaussian CDF to calculate probability temp falls in range
        lower = market.lower_bound
        upper = market.upper_bound

        if lower is None or upper is None:
            return 0.5, predicted_temp, {}

        # P(temp in bracket) = CDF(upper) - CDF(lower)
        prob = norm.cdf(upper, loc=predicted_temp, scale=temp_std) - \
               norm.cdf(lower, loc=predicted_temp, scale=temp_std)

    else:
        # THRESHOLD market: P(temp > threshold) or P(temp < threshold)
        threshold = market.threshold

        if threshold is None:
            return 0.5, predicted_temp, {}

        if market.comparison in ("above", "at_least"):
            # P(temp > threshold)
            prob = 1 - norm.cdf(threshold, loc=predicted_temp, scale=temp_std)
        else:
            # P(temp < threshold)
            prob = norm.cdf(threshold, loc=predicted_temp, scale=temp_std)

    # Clamp probability to reasonable bounds
    prob = max(0.01, min(0.99, prob))

    return prob, predicted_temp, {}


def calculate_calibrated_edge(
    market: WeatherMarket,
    use_fast_path: bool = False,
) -> Optional[CalibratedEdge]:
    """Calculate calibrated edge for a single weather market.

    Uses ensemble_v2 predictions and probabilistic calibration to
    compare against market-implied probability.

    Args:
        market: WeatherMarket to analyze
        use_fast_path: Use V3-first fast prediction (default False for accuracy)

    Returns:
        CalibratedEdge if edge can be calculated, None otherwise
    """
    # Check if we have required market info
    if market.lat is None or market.lon is None:
        return None

    if market.target_date is None:
        return None

    if market.condition_type == "unknown":
        return None

    # For bracket markets, we need bounds; for threshold markets, we need threshold
    if market.market_type == "bracket":
        if market.lower_bound is None or market.upper_bound is None:
            return None
    else:
        if market.threshold is None:
            return None

    # Get location string
    location = f"{market.location}, {market.state}" if market.state else market.location

    # Get ensemble prediction (fast path uses V3 regional model first)
    try:
        if use_fast_path:
            ensemble = predict_ensemble_fast(location, min_confidence=0.6, fallback_to_full=True)
        else:
            ensemble = predict_ensemble(location, include_nn=True)
    except Exception as e:
        print(f"Ensemble prediction error for {location}: {e}")
        return None

    if ensemble is None:
        return None

    # Get calibrated probability
    model_prob, predicted_temp, per_source_probs = _get_calibrated_probability(
        ensemble, market
    )

    if model_prob is None:
        return None

    # Get market implied probability
    # Use midpoint of bid/ask as best estimate
    if market.yes_bid > 0 and market.yes_ask > 0:
        kalshi_prob = (market.yes_bid + market.yes_ask) / 2
    elif market.last_price > 0:
        kalshi_prob = market.last_price
    else:
        return None

    # Avoid division issues
    if kalshi_prob < 0.01:
        kalshi_prob = 0.01
    if kalshi_prob > 0.99:
        kalshi_prob = 0.99

    # MARKET PRICE FILTER: Skip extreme odds markets (user wants 15-85% range)
    min_market_price = AGGRESSIVE_CONFIG["min_market_price"]
    max_market_price = AGGRESSIVE_CONFIG["max_market_price"]
    if kalshi_prob < min_market_price or kalshi_prob > max_market_price:
        # Market is at extreme odds (too certain either way) - skip
        return None

    # VALUE FILTER: Skip contracts with limited upside
    # Don't buy YES above 85¢ (only 15¢ max profit)
    # Don't buy NO below 15¢ (equivalent to YES > 85¢)
    max_yes_price = AGGRESSIVE_CONFIG["max_yes_price"]
    min_yes_price = AGGRESSIVE_CONFIG["min_yes_price"]

    # Determine side first based on model vs market comparison
    # For bracket markets, conviction thresholds are lower since bracket probs are smaller
    if market.market_type == "bracket":
        # For brackets: if model thinks this bracket is more likely than market, go YES
        # If model thinks this bracket is less likely than market, go NO
        # Bracket conviction: model prob should be at least 50% higher/lower than market
        if model_prob > kalshi_prob * 1.2:  # Model thinks 20%+ more likely
            side = "YES"
            edge_pct = (model_prob - kalshi_prob) / kalshi_prob * 100
        elif model_prob < kalshi_prob * 0.8:  # Model thinks 20%+ less likely
            side = "NO"
            model_no_prob = 1 - model_prob
            kalshi_no_prob = 1 - kalshi_prob
            edge_pct = (model_no_prob - kalshi_no_prob) / kalshi_no_prob * 100
        else:
            # No meaningful edge on this bracket
            return None
    else:
        # For threshold markets, use conviction thresholds
        MIN_YES_CONVICTION = 0.30  # Aggressive: model must think >= 30% YES
        MIN_NO_CONVICTION = 0.70   # Aggressive: model must think >= 70% NO

        if model_prob >= MIN_YES_CONVICTION and model_prob > kalshi_prob:
            side = "YES"
            edge_pct = (model_prob - kalshi_prob) / kalshi_prob * 100
        elif (1 - model_prob) >= MIN_NO_CONVICTION and model_prob < kalshi_prob:
            side = "NO"
            model_no_prob = 1 - model_prob
            kalshi_no_prob = 1 - kalshi_prob
            edge_pct = (model_no_prob - kalshi_no_prob) / kalshi_no_prob * 100
        else:
            return None

    # Apply value filter AFTER determining side
    if side == "YES" and kalshi_prob > max_yes_price:
        # YES contract too expensive (limited upside)
        return None
    if side == "NO" and kalshi_prob < min_yes_price:
        # NO contract too expensive (YES price too low)
        return None

    # Calculate expected value
    if side == "YES":
        cost = market.yes_ask
        payout = 1.0 - cost
        expected_value = model_prob * payout - (1 - model_prob) * cost
    else:
        cost = 1.0 - market.yes_bid
        payout = 1.0 - cost
        expected_value = (1 - model_prob) * payout - model_prob * cost

    # Calculate Kelly fraction
    kelly_fraction = _calculate_kelly_fraction(model_prob, kalshi_prob, side)

    # Get model agreement from ensemble confidence
    model_agreement = ensemble.confidence

    # Calculate confidence
    confidence, confidence_level = _calculate_confidence(
        market, model_prob, edge_pct, model_agreement, ensemble
    )

    # Get temperature std
    if market.condition_type == "temp_high":
        temp_std = ensemble.high_std
    elif market.condition_type == "temp_low":
        temp_std = ensemble.low_std
    else:
        temp_std = None

    # Generate explanation
    if predicted_temp is not None:
        temp_info = f"Ensemble predicts {predicted_temp:.1f}°F"
        if temp_std:
            temp_info += f" (±{temp_std:.1f}°F)"
    else:
        temp_info = f"Ensemble precip prob: {ensemble.precip_probability:.0f}%"

    # Generate condition description based on market type
    if market.market_type == "bracket":
        condition_desc = f"{market.lower_bound}-{market.upper_bound}°F"
    else:
        condition_desc = {
            "temp_high": f"high >{market.threshold}°F",
            "temp_low": f"low <{market.threshold}°F",
            "rain": "rain",
            "snow": "snow",
        }.get(market.condition_type, market.condition_type)

    explanation = (
        f"{temp_info}. "
        f"P({condition_desc}) = {model_prob*100:.1f}%, "
        f"market implies {kalshi_prob*100:.1f}%. "
        f"Edge: {edge_pct:+.1f}% ({side}). "
        f"Kelly: {kelly_fraction:.1%} of bankroll."
    )

    return CalibratedEdge(
        market=market,
        kalshi_prob=kalshi_prob,
        model_prob=model_prob,
        edge_pct=edge_pct,
        expected_value=expected_value,
        kelly_fraction=kelly_fraction,
        confidence=confidence,
        confidence_level=confidence_level,
        side=side,
        explanation=explanation,
        ensemble_prediction=ensemble,
        model_agreement=model_agreement,
        sources_used=ensemble.sources_used,
        predicted_temp=predicted_temp,
        temp_std=temp_std,
        per_source_probs=per_source_probs,
    )


def find_calibrated_edges(
    min_edge: float = 10.0,
    min_confidence: float = 0.0,
    min_kelly: float = 0.0,
    max_series: int = 100,
    days_ahead: int = 7,
    use_fast_path: bool = False,
) -> List[CalibratedEdge]:
    """Find all calibrated edge opportunities above thresholds.

    Scans weather markets and calculates calibrated edges for each,
    returning those that meet the minimum criteria.

    Args:
        min_edge: Minimum absolute edge percentage (default 10%)
        min_confidence: Minimum confidence score 0-1 (default 0, no filter)
        min_kelly: Minimum Kelly fraction (default 0, no filter)
        max_series: Maximum number of series to query from Kalshi
        days_ahead: Only include markets expiring within this many days
        use_fast_path: Use V3-first fast prediction for speed (default False)

    Returns:
        List of CalibratedEdge objects sorted by Kelly * confidence (best first)

    Example:
        >>> edges = find_calibrated_edges(min_edge=15, min_confidence=0.5)
        >>> for e in edges[:5]:
        ...     print(f"{e.market.ticker}: {e.edge_pct:+.1f}% edge, Kelly: {e.kelly_fraction:.1%}")
    """
    # Scan markets
    mode = "FAST" if use_fast_path else "FULL"
    print(f"Scanning Kalshi weather markets ({mode} mode, max {max_series} series, {days_ahead} days ahead)...")
    markets = scan_weather_markets(
        max_series=max_series,
        days_ahead=days_ahead,
    )
    print(f"Found {len(markets)} weather markets")

    # Calculate calibrated edges
    edges = []
    processed = 0

    for market in markets:
        processed += 1
        if processed % 10 == 0:
            print(f"  Processing market {processed}/{len(markets)}...")

        edge = calculate_calibrated_edge(market, use_fast_path=use_fast_path)

        if edge is None:
            continue

        # Apply filters
        if abs(edge.edge_pct) < min_edge:
            continue
        if edge.confidence < min_confidence:
            continue
        if edge.kelly_fraction < min_kelly:
            continue

        edges.append(edge)

    # Sort by Kelly * confidence (best opportunities first)
    # This prioritizes edges that are both high-confidence and have good bet sizing
    edges.sort(key=lambda e: e.kelly_fraction * e.confidence, reverse=True)

    print(f"Found {len(edges)} calibrated edges meeting criteria")

    return edges


def format_calibrated_edge(edge: CalibratedEdge) -> str:
    """Format a CalibratedEdge for display.

    Args:
        edge: CalibratedEdge to format

    Returns:
        Formatted string representation
    """
    lines = [
        f"Market: {edge.market.ticker}",
        f"  {edge.market.title}",
        f"  Location: {edge.market.location}, {edge.market.state or 'N/A'}",
        f"  Target Date: {edge.market.target_date}",
        f"  Condition: {edge.market.condition_type} ({edge.market.comparison} {edge.market.threshold})",
        f"",
        f"  Market Probability: {edge.kalshi_prob*100:.1f}%",
        f"  Calibrated Model:   {edge.model_prob*100:.1f}%",
        f"  Edge:               {edge.edge_pct:+.1f}%",
        f"",
        f"  Recommended Side:   {edge.side}",
        f"  Expected Value:     ${edge.expected_value:.3f} per $1",
        f"  Kelly Fraction:     {edge.kelly_fraction:.1%} of bankroll",
        f"  Confidence:         {edge.confidence:.0%} ({edge.confidence_level})",
        f"",
    ]

    if edge.predicted_temp is not None:
        temp_line = f"  Ensemble Temp:      {edge.predicted_temp:.1f}°F"
        if edge.temp_std:
            temp_line += f" (±{edge.temp_std:.1f}°F)"
        lines.append(temp_line)

    if edge.sources_used:
        lines.append(f"  Sources Used:       {', '.join(edge.sources_used)}")

    lines.extend([
        f"  Model Agreement:    {edge.model_agreement:.0%}",
        f"",
        f"  {edge.explanation}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Calibrated Edge Calculator V2[/bold cyan]\n"
        "Using ensemble predictions + probabilistic calibration",
        border_style="cyan"
    ))

    # Find edges with moderate threshold for testing
    console.print("\n[bold]Scanning for calibrated edges...[/bold]\n")

    edges = find_calibrated_edges(min_edge=5, min_confidence=0.3)

    if not edges:
        console.print("[yellow]No calibrated edge opportunities found above threshold[/yellow]")
        console.print("This could mean:")
        console.print("  - Markets are efficiently priced")
        console.print("  - Calibrated probabilities agree with market")
        console.print("  - No weather markets match our model's locations")
    else:
        console.print(f"\n[green]Found {len(edges)} calibrated edges:[/green]\n")

        table = Table(title="Calibrated Edge Opportunities")
        table.add_column("Ticker", style="cyan", max_width=22)
        table.add_column("Location", max_width=12)
        table.add_column("Date")
        table.add_column("Mkt", justify="right")
        table.add_column("Model", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("Kelly", justify="right")
        table.add_column("Side")
        table.add_column("Conf")

        for edge in edges[:15]:  # Show top 15
            edge_style = "green" if edge.edge_pct > 0 else "red"
            side_style = "green" if edge.side == "YES" else "red"
            conf_style = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(edge.confidence_level, "white")

            table.add_row(
                edge.market.ticker[:22],
                edge.market.location[:12],
                str(edge.market.target_date) if edge.market.target_date else "N/A",
                f"{edge.kalshi_prob*100:.0f}%",
                f"{edge.model_prob*100:.0f}%",
                f"[{edge_style}]{edge.edge_pct:+.0f}%[/{edge_style}]",
                f"{edge.kelly_fraction:.1%}",
                f"[{side_style}]{edge.side}[/{side_style}]",
                f"[{conf_style}]{edge.confidence_level}[/{conf_style}]",
            )

        console.print(table)

        if len(edges) > 15:
            console.print(f"\n... and {len(edges) - 15} more opportunities")

        # Show top opportunity details
        if edges:
            console.print("\n[bold]Top Opportunity Details:[/bold]")
            console.print(Panel(format_calibrated_edge(edges[0]), border_style="green"))
