"""Oracle-powered edge calculator.

This edge calculator uses the Weather Oracle (trained on actual outcomes)
instead of forecast-based predictions. It learns from 37K+ real Kalshi
market outcomes to predict what will actually happen.

Key difference from edge.py and edge_v2.py:
- Those use weather forecasts to estimate probability
- This uses the Oracle model trained on actual outcomes
- The Oracle knows patterns like "extreme thresholds rarely hit"
"""

import datetime
from dataclasses import dataclass
from typing import Optional, List

from src.kalshi.scanner import WeatherMarket, scan_weather_markets
from src.models.oracle_v2 import WeatherOracleV2


@dataclass
class OracleEdge:
    """Edge opportunity identified by the Oracle."""
    market: WeatherMarket
    kalshi_prob: float
    oracle_prob: float
    edge_pct: float
    expected_value: float
    side: str  # "YES" or "NO"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    explanation: str

    # Oracle-specific info
    seasonal_normal: float
    threshold_deviation: float
    is_extreme: bool


def _calculate_edge(
    oracle_prob: float,
    kalshi_prob: float,
    side: str,
    yes_ask: float,
    yes_bid: float,
) -> tuple[float, float]:
    """Calculate edge percentage and expected value.

    Args:
        oracle_prob: Oracle's probability of YES
        kalshi_prob: Market's implied probability
        side: "YES" or "NO"
        yes_ask: Price to buy YES
        yes_bid: Price to sell YES (buy NO)

    Returns:
        Tuple of (edge_pct, expected_value)
    """
    if side == "YES":
        # Edge for YES bet
        edge_pct = (oracle_prob - kalshi_prob) / kalshi_prob * 100
        cost = yes_ask
        payout = 1.0 - cost
        expected_value = oracle_prob * payout - (1 - oracle_prob) * cost
    else:
        # Edge for NO bet
        oracle_no_prob = 1 - oracle_prob
        kalshi_no_prob = 1 - kalshi_prob
        edge_pct = (oracle_no_prob - kalshi_no_prob) / kalshi_no_prob * 100
        cost = 1.0 - yes_bid
        payout = 1.0 - cost
        expected_value = oracle_no_prob * payout - oracle_prob * cost

    return edge_pct, expected_value


def _determine_confidence(
    edge_pct: float,
    oracle_prob: float,
    threshold_deviation: float,
    market_volume: int,
) -> str:
    """Determine confidence level for the edge.

    Higher confidence when:
    - Edge is large
    - Oracle probability is not extreme
    - Threshold deviation is moderate (well within training distribution)
    - Market has decent volume (prices are meaningful)
    """
    score = 0

    # Edge size
    if abs(edge_pct) > 30:
        score += 2
    elif abs(edge_pct) > 15:
        score += 1

    # Oracle probability reasonableness
    if 0.2 <= oracle_prob <= 0.8:
        score += 2
    elif 0.1 <= oracle_prob <= 0.9:
        score += 1

    # Threshold deviation (extreme = less confident)
    if abs(threshold_deviation) < 1:
        score += 2
    elif abs(threshold_deviation) < 2:
        score += 1

    # Market volume
    if market_volume > 100:
        score += 1

    if score >= 5:
        return "HIGH"
    elif score >= 3:
        return "MEDIUM"
    return "LOW"


def find_oracle_edges(
    min_edge: float = 15.0,
    max_series: int = 100,
    days_ahead: int = 7,
    oracle: Optional[WeatherOracleV2] = None,
) -> List[OracleEdge]:
    """Find edge opportunities using the Oracle model.

    Args:
        min_edge: Minimum absolute edge percentage to include
        max_series: Maximum Kalshi series to query
        days_ahead: Only include markets expiring within this many days
        oracle: Optional Oracle instance (creates new one if not provided)

    Returns:
        List of OracleEdge opportunities sorted by edge size
    """
    # Initialize Oracle
    if oracle is None:
        oracle = WeatherOracleV2()

    # Conviction thresholds (same as edge.py)
    MIN_YES_CONVICTION = 0.35
    MIN_NO_CONVICTION = 0.65

    # Scan markets
    print("Scanning Kalshi weather markets...")
    markets = scan_weather_markets(max_series=max_series, days_ahead=days_ahead)
    print(f"Found {len(markets)} markets")

    edges = []
    processed = 0

    for market in markets:
        processed += 1
        if processed % 20 == 0:
            print(f"  Processing {processed}/{len(markets)}...")

        # Skip unsupported market types
        if market.condition_type not in ("temp_high", "temp_low"):
            continue

        if market.location == "Unknown" or market.threshold is None:
            continue

        if market.target_date is None:
            continue

        # Get Oracle prediction
        try:
            oracle_prob, explanation = oracle.predict(
                location=market.location,
                condition_type=market.condition_type,
                comparison=market.comparison or "above",
                threshold=market.threshold,
                target_date=market.target_date,
            )
        except Exception as e:
            continue

        # Get market probability
        if market.yes_bid > 0 and market.yes_ask > 0:
            kalshi_prob = (market.yes_bid + market.yes_ask) / 2
        elif market.last_price > 0:
            kalshi_prob = market.last_price
        else:
            continue

        # Avoid edge cases
        if kalshi_prob < 0.01:
            kalshi_prob = 0.01
        if kalshi_prob > 0.99:
            kalshi_prob = 0.99

        # Determine side based on conviction (same logic as edge.py)
        if oracle_prob >= MIN_YES_CONVICTION and oracle_prob > kalshi_prob:
            side = "YES"
            edge_pct, ev = _calculate_edge(
                oracle_prob, kalshi_prob, "YES",
                market.yes_ask, market.yes_bid
            )
        elif (1 - oracle_prob) >= MIN_NO_CONVICTION and oracle_prob < kalshi_prob:
            side = "NO"
            edge_pct, ev = _calculate_edge(
                oracle_prob, kalshi_prob, "NO",
                market.yes_ask, market.yes_bid
            )
        else:
            # No clear edge - Oracle lacks conviction
            continue

        # Apply minimum edge filter
        if abs(edge_pct) < min_edge:
            continue

        # Determine confidence
        confidence = _determine_confidence(
            edge_pct,
            oracle_prob,
            explanation["threshold_deviation"],
            market.volume,
        )

        # Build explanation
        dev = explanation["threshold_deviation"]
        normal = explanation["seasonal_normal"]
        exp_text = (
            f"Oracle predicts {oracle_prob*100:.1f}% for {market.location} "
            f"{market.condition_type} {market.comparison} {market.threshold}°F. "
            f"Seasonal normal is {normal}°F ({dev:+.1f}σ from threshold). "
            f"Market implies {kalshi_prob*100:.1f}%. "
            f"Recommend {side}."
        )

        edges.append(OracleEdge(
            market=market,
            kalshi_prob=kalshi_prob,
            oracle_prob=oracle_prob,
            edge_pct=edge_pct,
            expected_value=ev,
            side=side,
            confidence=confidence,
            explanation=exp_text,
            seasonal_normal=normal,
            threshold_deviation=dev,
            is_extreme=explanation["is_extreme"],
        ))

    # Sort by absolute edge
    edges.sort(key=lambda e: abs(e.edge_pct), reverse=True)

    print(f"Found {len(edges)} Oracle edges meeting criteria")
    return edges


def format_oracle_edge(edge: OracleEdge) -> str:
    """Format an OracleEdge for display."""
    lines = [
        f"🔮 ORACLE EDGE: {edge.market.ticker}",
        f"   {edge.market.title}",
        f"   Location: {edge.market.location} | Date: {edge.market.target_date}",
        f"   Threshold: {edge.market.threshold}°F ({edge.market.comparison})",
        f"   ---",
        f"   Kalshi:  {edge.kalshi_prob*100:.1f}%",
        f"   Oracle:  {edge.oracle_prob*100:.1f}%",
        f"   Edge:    {edge.edge_pct:+.1f}%",
        f"   EV:      ${edge.expected_value:.3f}/contract",
        f"   ---",
        f"   Side:       {edge.side}",
        f"   Confidence: {edge.confidence}",
        f"   Normal:     {edge.seasonal_normal}°F (deviation: {edge.threshold_deviation:+.1f}σ)",
        f"   ---",
        f"   {edge.explanation}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]🔮 Oracle Edge Calculator[/bold cyan]\n"
        "Finding edges using the Weather Oracle\n"
        "[dim]Trained on 37K+ real Kalshi outcomes[/dim]",
        border_style="cyan"
    ))

    # Find edges
    edges = find_oracle_edges(min_edge=10, days_ahead=5)

    if not edges:
        console.print("\n[yellow]No Oracle edges found above threshold[/yellow]")
    else:
        console.print(f"\n[green]Found {len(edges)} Oracle edges![/green]\n")

        table = Table(title="Oracle Edge Opportunities")
        table.add_column("Ticker", style="cyan", max_width=25)
        table.add_column("Location")
        table.add_column("Threshold")
        table.add_column("Kalshi", justify="right")
        table.add_column("Oracle", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("Side")
        table.add_column("Conf")

        for edge in edges[:15]:
            edge_style = "green" if edge.edge_pct > 0 else "red"
            side_style = "green" if edge.side == "YES" else "red"
            conf_style = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(edge.confidence, "white")

            table.add_row(
                edge.market.ticker[:25],
                edge.market.location[:10],
                f"{edge.market.threshold}°F",
                f"{edge.kalshi_prob*100:.0f}%",
                f"{edge.oracle_prob*100:.0f}%",
                f"[{edge_style}]{edge.edge_pct:+.0f}%[/{edge_style}]",
                f"[{side_style}]{edge.side}[/{side_style}]",
                f"[{conf_style}]{edge.confidence}[/{conf_style}]",
            )

        console.print(table)

        # Show top edge details
        if edges:
            console.print("\n[bold]Top Oracle Edge:[/bold]")
            console.print(Panel(format_oracle_edge(edges[0]), border_style="green"))
