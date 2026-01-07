"""Edge calculator comparing model predictions vs Kalshi market prices.

This module calculates the "edge" - the difference between our model's
predicted probability and the market's implied probability. A positive
edge indicates the market may be undervaluing the event.
"""

import datetime
from dataclasses import dataclass
from typing import Optional

from src.inference.ensemble import EnsemblePredictor, EnsembleForecast
from src.kalshi.scanner import WeatherMarket, scan_weather_markets


@dataclass
class EdgeOpportunity:
    """Represents an edge opportunity where model disagrees with market.

    Attributes:
        market: The WeatherMarket this opportunity is for
        kalshi_prob: Market's implied probability (0-1)
        model_prob: Model's predicted probability (0-1)
        edge_pct: Edge percentage: (model_prob - kalshi_prob) / kalshi_prob * 100
        expected_value: Expected value of a $1 bet: model_prob * payout - (1 - model_prob) * cost
        confidence: Confidence level based on model uncertainty and market liquidity
        side: "YES" if we think probability is higher than market, "NO" otherwise
        explanation: Text explaining the edge
    """
    market: WeatherMarket
    kalshi_prob: float
    model_prob: float
    edge_pct: float
    expected_value: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    side: str  # "YES" or "NO"
    explanation: str


def _get_model_temp_probability(
    forecast: EnsembleForecast,
    target_date: datetime.date,
    threshold: float,
    comparison: str,
    condition_type: str,
) -> Optional[float]:
    """Calculate probability of temperature condition from model forecast.

    Args:
        forecast: Ensemble forecast with hourly predictions
        target_date: Date the market is about
        threshold: Temperature threshold in Fahrenheit
        comparison: "above" or "below"
        condition_type: "temp_high" or "temp_low"

    Returns:
        Probability (0-1) or None if can't determine
    """
    if not forecast or not forecast.hourly:
        return None

    # Get predictions for the target date
    target_predictions = []
    for h in forecast.hourly:
        if h.timestamp.date() == target_date:
            target_predictions.append(h)

    if not target_predictions:
        # Use all predictions if target date is beyond forecast range
        target_predictions = forecast.hourly

    if not target_predictions:
        return None

    # For high temp markets, look at max temperature of the day
    # For low temp markets, look at min temperature
    if condition_type == "temp_high":
        relevant_temp = max(h.temperature for h in target_predictions)
    elif condition_type == "temp_low":
        relevant_temp = min(h.temperature for h in target_predictions)
    else:
        # For unknown, use average
        relevant_temp = sum(h.temperature for h in target_predictions) / len(target_predictions)

    # Calculate probability based on comparison
    # Using a simple approach: if predicted temp is far from threshold,
    # probability is high; if close, probability is around 50%

    diff = relevant_temp - threshold  # Positive if temp > threshold

    # Use sigmoid-like probability based on difference
    # Each degree away from threshold adds/subtracts ~12% probability
    # Clamp between 5% and 95%
    import math

    if comparison in ("above", "at_least"):
        # Probability that temp >= threshold
        # Higher predicted temp -> higher probability
        prob = 1 / (1 + math.exp(-diff / 3))  # Sigmoid with scale factor
    elif comparison in ("below", "at_most"):
        # Probability that temp < threshold
        # Lower predicted temp -> higher probability
        prob = 1 / (1 + math.exp(diff / 3))
    else:
        prob = 0.5

    # Clamp to reasonable bounds
    prob = max(0.05, min(0.95, prob))

    return prob


def _get_model_precip_probability(
    forecast: EnsembleForecast,
    target_date: datetime.date,
    threshold: float,
    comparison: str,
) -> Optional[float]:
    """Calculate probability of precipitation condition from model forecast.

    Args:
        forecast: Ensemble forecast with hourly predictions
        target_date: Date the market is about
        threshold: Precipitation threshold (inches or probability)
        comparison: "above", "below", "at_least", "at_most"

    Returns:
        Probability (0-1) or None if can't determine
    """
    if not forecast or not forecast.hourly:
        return None

    # Get predictions for the target date
    target_predictions = []
    for h in forecast.hourly:
        if h.timestamp.date() == target_date:
            target_predictions.append(h)

    if not target_predictions:
        target_predictions = forecast.hourly

    if not target_predictions:
        return None

    # For rain markets, look at max precipitation probability during the day
    max_precip_prob = max(h.precip_probability for h in target_predictions)

    # Convert from 0-100 scale to 0-1
    precip_prob = max_precip_prob / 100.0

    # For "will it rain" type markets, use precip probability directly
    # Adjust based on threshold if provided
    if threshold is not None:
        # Threshold might be in inches - convert to probability estimate
        if threshold <= 0.1:
            # Any measurable precipitation
            return precip_prob
        else:
            # Higher threshold = less likely
            return max(0.05, precip_prob * 0.7)

    return precip_prob


def _calculate_confidence(
    market: WeatherMarket,
    model_prob: float,
    edge_pct: float,
) -> str:
    """Calculate confidence level for an edge opportunity.

    Args:
        market: The weather market
        model_prob: Model's predicted probability
        edge_pct: Calculated edge percentage

    Returns:
        "HIGH", "MEDIUM", or "LOW"
    """
    confidence_score = 0

    # Higher volume = more confidence in market efficiency (lower confidence in our edge)
    # But also means better liquidity for trading
    if market.volume > 1000:
        confidence_score += 1
    elif market.volume > 100:
        confidence_score += 2
    else:
        confidence_score += 0  # Low volume might mean inefficient market

    # Larger edge = more confidence
    if abs(edge_pct) > 30:
        confidence_score += 2
    elif abs(edge_pct) > 15:
        confidence_score += 1

    # Model probability not extreme = more confidence
    if 0.2 <= model_prob <= 0.8:
        confidence_score += 1

    # Good bid/ask spread = more confidence
    spread = market.yes_ask - market.yes_bid
    if spread <= 0.05:
        confidence_score += 1

    if confidence_score >= 4:
        return "HIGH"
    elif confidence_score >= 2:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_edge(
    market: WeatherMarket,
    predictor: Optional[EnsemblePredictor] = None,
) -> Optional[EdgeOpportunity]:
    """Calculate edge for a single weather market.

    Compares Kalshi's market price (implied probability) to our model's
    predicted probability for the weather event.

    Args:
        market: WeatherMarket to analyze
        predictor: Optional EnsemblePredictor instance. Creates new one if not provided.

    Returns:
        EdgeOpportunity if edge can be calculated, None otherwise

    Example:
        >>> from src.kalshi.scanner import scan_weather_markets
        >>> markets = scan_weather_markets()
        >>> if markets:
        ...     edge = calculate_edge(markets[0])
        ...     print(f"Edge: {edge.edge_pct:.1f}%")
    """
    # Initialize predictor if needed
    if predictor is None:
        try:
            predictor = EnsemblePredictor()
        except Exception:
            return None

    # Check if we have the required market info
    if market.lat is None or market.lon is None:
        return None

    if market.target_date is None:
        return None

    if market.condition_type == "unknown":
        return None

    # Get ensemble forecast for this location
    try:
        forecast = predictor.get_ensemble_forecast(
            market.lat,
            market.lon,
            location_name=f"{market.location}, {market.state}" if market.state else market.location,
        )
    except Exception:
        return None

    if forecast is None:
        return None

    # Calculate model probability based on condition type
    model_prob = None

    if market.condition_type in ("temp_high", "temp_low"):
        if market.threshold is not None and market.comparison is not None:
            model_prob = _get_model_temp_probability(
                forecast,
                market.target_date,
                market.threshold,
                market.comparison,
                market.condition_type,
            )
    elif market.condition_type in ("rain", "snow"):
        model_prob = _get_model_precip_probability(
            forecast,
            market.target_date,
            market.threshold,
            market.comparison or "at_least",
        )

    if model_prob is None:
        return None

    # Get market implied probability
    # Use midpoint of bid/ask as best estimate, or last price if available
    if market.yes_bid > 0 and market.yes_ask > 0:
        kalshi_prob = (market.yes_bid + market.yes_ask) / 2
    elif market.last_price > 0:
        kalshi_prob = market.last_price
    else:
        return None

    # Avoid division by zero
    if kalshi_prob < 0.01:
        kalshi_prob = 0.01

    # Calculate edge
    # edge_pct = (model_prob - kalshi_prob) / kalshi_prob * 100
    edge_pct = (model_prob - kalshi_prob) / kalshi_prob * 100

    # Determine side: if model thinks probability is higher, bet YES
    side = "YES" if model_prob > kalshi_prob else "NO"

    # Calculate expected value
    # For YES bet at price P: EV = model_prob * (1 - P) - (1 - model_prob) * P
    # Simplified: EV = model_prob - P
    if side == "YES":
        cost = market.yes_ask  # Buy YES at ask price
        payout = 1.0 - cost  # Potential profit if YES
        expected_value = model_prob * payout - (1 - model_prob) * cost
    else:
        # For NO bet: buy NO, which means selling YES
        cost = 1.0 - market.yes_bid  # Buy NO at (1 - bid)
        payout = 1.0 - cost  # Potential profit if NO
        expected_value = (1 - model_prob) * payout - model_prob * cost

    # Calculate confidence
    confidence = _calculate_confidence(market, model_prob, edge_pct)

    # Generate explanation
    if market.condition_type == "temp_high":
        condition_desc = f"high temp {market.comparison} {market.threshold}°F"
    elif market.condition_type == "temp_low":
        condition_desc = f"low temp {market.comparison} {market.threshold}°F"
    elif market.condition_type == "rain":
        condition_desc = "rain"
    elif market.condition_type == "snow":
        condition_desc = "snow"
    else:
        condition_desc = market.condition_type

    explanation = (
        f"Model predicts {model_prob*100:.1f}% chance of {condition_desc} "
        f"in {market.location} on {market.target_date}, "
        f"while market implies {kalshi_prob*100:.1f}%. "
        f"Consider {side} position."
    )

    return EdgeOpportunity(
        market=market,
        kalshi_prob=kalshi_prob,
        model_prob=model_prob,
        edge_pct=edge_pct,
        expected_value=expected_value,
        confidence=confidence,
        side=side,
        explanation=explanation,
    )


def find_edges(
    min_edge: float = 10,
    max_series: int = 100,
    days_ahead: int = 7,
) -> list[EdgeOpportunity]:
    """Find all edge opportunities above the minimum threshold.

    Scans weather markets and calculates edges for each, returning
    those where the edge exceeds the minimum threshold.

    Args:
        min_edge: Minimum absolute edge percentage to include (default 10%)
        max_series: Maximum number of series to query from Kalshi
        days_ahead: Only include markets expiring within this many days

    Returns:
        List of EdgeOpportunity objects sorted by absolute edge (highest first)

    Example:
        >>> edges = find_edges(min_edge=15)
        >>> for e in edges[:5]:
        ...     print(f"{e.market.ticker}: {e.edge_pct:+.1f}% edge ({e.side})")
    """
    # Initialize predictor once for efficiency
    try:
        predictor = EnsemblePredictor()
    except Exception:
        return []

    # Scan markets
    markets = scan_weather_markets(
        max_series=max_series,
        days_ahead=days_ahead,
    )

    # Calculate edges
    edges = []
    for market in markets:
        edge = calculate_edge(market, predictor)
        if edge is not None and abs(edge.edge_pct) >= min_edge:
            edges.append(edge)

    # Sort by absolute edge (highest first)
    edges.sort(key=lambda e: abs(e.edge_pct), reverse=True)

    return edges


def format_edge_opportunity(edge: EdgeOpportunity) -> str:
    """Format an EdgeOpportunity for display.

    Args:
        edge: EdgeOpportunity to format

    Returns:
        Formatted string representation
    """
    lines = [
        f"Market: {edge.market.ticker}",
        f"  {edge.market.title}",
        f"  Location: {edge.market.location}, {edge.market.state or 'N/A'}",
        f"  Target Date: {edge.market.target_date}",
        f"  Condition: {edge.market.condition_type}",
        f"  Threshold: {edge.market.threshold} ({edge.market.comparison})" if edge.market.threshold else "",
        f"  ---",
        f"  Kalshi Probability: {edge.kalshi_prob*100:.1f}%",
        f"  Model Probability:  {edge.model_prob*100:.1f}%",
        f"  Edge: {edge.edge_pct:+.1f}%",
        f"  Expected Value: ${edge.expected_value:.3f} per $1 bet",
        f"  Recommended: {edge.side}",
        f"  Confidence: {edge.confidence}",
        f"  ---",
        f"  {edge.explanation}",
    ]
    return "\n".join(line for line in lines if line)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]Finding Edge Opportunities[/bold cyan]\n")

    # Find edges with low threshold for testing
    console.print("Scanning markets and calculating edges (min 5% edge)...")

    edges = find_edges(min_edge=5)

    if not edges:
        console.print("[yellow]No edge opportunities found above threshold[/yellow]")
        console.print("This could mean:")
        console.print("  - Markets are efficiently priced")
        console.print("  - No weather markets match our model's locations")
        console.print("  - API rate limits prevented fetching all data")
    else:
        console.print(f"\n[green]Found {len(edges)} edge opportunities:[/green]\n")

        table = Table(title="Edge Opportunities")
        table.add_column("Ticker", style="cyan")
        table.add_column("Location")
        table.add_column("Date")
        table.add_column("Kalshi", justify="right")
        table.add_column("Model", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("EV", justify="right")
        table.add_column("Side", justify="center")
        table.add_column("Conf")

        for edge in edges[:10]:  # Show top 10
            edge_style = "green" if edge.edge_pct > 0 else "red"
            table.add_row(
                edge.market.ticker[:20],
                f"{edge.market.location}",
                str(edge.market.target_date) if edge.market.target_date else "N/A",
                f"{edge.kalshi_prob*100:.0f}%",
                f"{edge.model_prob*100:.0f}%",
                f"[{edge_style}]{edge.edge_pct:+.0f}%[/{edge_style}]",
                f"${edge.expected_value:.2f}",
                f"[{'green' if edge.side == 'YES' else 'red'}]{edge.side}[/]",
                edge.confidence,
            )

        console.print(table)

        if len(edges) > 10:
            console.print(f"\n... and {len(edges) - 10} more opportunities")
