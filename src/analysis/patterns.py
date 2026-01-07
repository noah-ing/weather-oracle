"""Pattern recognition for high-confidence weather situations.

This module identifies weather patterns where forecast models historically
agree or disagree, helping to flag which situations are good or bad for betting.

High-confidence situations:
- All models agree (low spread)
- Stable weather pattern (consistent forecasts)
- Good historical accuracy for this pattern

Uncertain situations:
- Models disagree significantly
- Frontal passage or storm system
- Edge cases (near threshold)

Usage:
    from src.analysis.patterns import classify_pattern

    result = classify_pattern("NYC", "2026-01-08")
    print(f"Confidence: {result.confidence_level}")
    print(f"Pattern: {result.pattern_type}")
    print(f"Recommendation: {result.recommendation}")
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from src.config import DB_PATH
from src.db.database import get_connection


@dataclass
class PatternClassification:
    """Classification of a weather pattern for a location and date.

    Attributes:
        location: City name (e.g., "New York, NY")
        target_date: Date being analyzed
        analysis_time: When this classification was made

        # Model agreement metrics
        model_spread_high: Std dev of high temp predictions across models (F)
        model_spread_low: Std dev of low temp predictions across models (F)
        models_used: Number of models with forecasts
        model_agreement_score: 0-1 score (1 = perfect agreement)

        # Pattern classification
        pattern_type: "stable", "transitional", "uncertain", "extreme"
        confidence_level: "HIGH", "MEDIUM", "LOW", "VERY_LOW"
        confidence_score: 0-100 numeric confidence

        # Situation flags
        is_frontal_passage: Whether a front is expected
        is_near_threshold: Whether temp is near market thresholds
        is_extreme_temp: Whether temps are unusual for location/season

        # Historical context
        historical_accuracy: Past accuracy for similar patterns
        similar_pattern_count: How many similar patterns in history

        # Recommendations
        recommendation: "BET", "CAUTION", "AVOID"
        explanation: Detailed explanation of classification
    """
    location: str
    target_date: date
    analysis_time: datetime

    # Model agreement
    model_spread_high: float
    model_spread_low: float
    models_used: int
    model_agreement_score: float

    # Classification
    pattern_type: str
    confidence_level: str
    confidence_score: float

    # Situation flags
    is_frontal_passage: bool
    is_near_threshold: bool
    is_extreme_temp: bool

    # Historical context
    historical_accuracy: Optional[float]
    similar_pattern_count: int

    # Recommendations
    recommendation: str
    explanation: str


def init_pattern_classifications_table():
    """Create the pattern_classifications table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pattern_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            target_date TEXT NOT NULL,
            analysis_time TEXT NOT NULL,
            model_spread_high REAL,
            model_spread_low REAL,
            models_used INTEGER,
            model_agreement_score REAL,
            pattern_type TEXT,
            confidence_level TEXT,
            confidence_score REAL,
            is_frontal_passage INTEGER,
            is_near_threshold INTEGER,
            is_extreme_temp INTEGER,
            historical_accuracy REAL,
            similar_pattern_count INTEGER,
            recommendation TEXT,
            explanation TEXT,
            UNIQUE(location, target_date, analysis_time)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pattern_location_date
        ON pattern_classifications(location, target_date)
    """)

    conn.commit()
    conn.close()


def _normalize_location(location: str) -> str:
    """Normalize location name for consistent lookups."""
    location_map = {
        "NYC": "New York, NY",
        "LA": "Los Angeles, CA",
        "SF": "San Francisco, CA",
        "CHI": "Chicago, IL",
        "DC": "Washington, DC",
        "PHILLY": "Philadelphia, PA",
        "MIAMI": "Miami, FL",
        "DENVER": "Denver, CO",
        "PHOENIX": "Phoenix, AZ",
        "SEATTLE": "Seattle, WA",
        "BOSTON": "Boston, MA",
        "ATLANTA": "Atlanta, GA",
        "DALLAS": "Dallas, TX",
        "HOUSTON": "Houston, TX",
    }

    upper = location.upper().strip()
    if upper in location_map:
        return location_map[upper]
    return location


def _get_ensemble_data(
    location: str,
    target_date: date,
) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """Get ensemble forecast data for a location and date.

    Returns:
        Tuple of (high_temps dict, low_temps dict, sources list)
    """
    # Try to get from forecast_log if we have it
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='forecast_log'
    """)

    if cursor.fetchone() is None:
        conn.close()
        return {}, {}, []

    # Get forecasts for this location and date
    location_normalized = _normalize_location(location)

    cursor.execute("""
        SELECT source, predicted_high, predicted_low
        FROM forecast_log
        WHERE location = ? AND target_date = ?
    """, (location_normalized, target_date.isoformat()))

    rows = cursor.fetchall()
    conn.close()

    high_temps = {}
    low_temps = {}
    sources = []

    for row in rows:
        source = row["source"]
        high_temps[source] = row["predicted_high"]
        low_temps[source] = row["predicted_low"]
        sources.append(source)

    return high_temps, low_temps, sources


def _get_live_ensemble(location: str) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """Get live ensemble data from weather APIs.

    Returns:
        Tuple of (high_temps dict, low_temps dict, sources list)
    """
    try:
        from src.api.weather_models import get_all_forecasts
        from src.api.geocoding import get_coordinates

        location_normalized = _normalize_location(location)
        parts = location_normalized.split(",")
        city = parts[0].strip()
        state = parts[1].strip() if len(parts) > 1 else None

        coords = get_coordinates(city, state)
        if coords is None:
            return {}, {}, []

        lat, lon = coords

        forecasts = get_all_forecasts(lat, lon)

        high_temps = {src: fc.high_temp for src, fc in forecasts.items()}
        low_temps = {src: fc.low_temp for src, fc in forecasts.items()}
        sources = list(forecasts.keys())

        return high_temps, low_temps, sources

    except Exception as e:
        print(f"Error getting live ensemble: {e}")
        return {}, {}, []


def _calculate_model_agreement(
    high_temps: Dict[str, float],
    low_temps: Dict[str, float],
) -> Tuple[float, float, float]:
    """Calculate model agreement metrics.

    Returns:
        Tuple of (spread_high, spread_low, agreement_score)
    """
    import statistics

    if len(high_temps) < 2:
        return 0.0, 0.0, 1.0  # Single model = perfect "agreement"

    highs = list(high_temps.values())
    lows = list(low_temps.values())

    spread_high = statistics.stdev(highs)
    spread_low = statistics.stdev(lows)

    # Agreement score: 0-1, where 1 is perfect agreement
    # Based on typical forecast spread ranges
    # < 2F spread = high agreement, > 5F = low agreement
    avg_spread = (spread_high + spread_low) / 2

    if avg_spread <= 1.5:
        agreement = 1.0
    elif avg_spread <= 3.0:
        agreement = 0.8 - (avg_spread - 1.5) * 0.2
    elif avg_spread <= 5.0:
        agreement = 0.5 - (avg_spread - 3.0) * 0.1
    else:
        agreement = max(0.1, 0.3 - (avg_spread - 5.0) * 0.05)

    return spread_high, spread_low, agreement


def _detect_frontal_passage(
    high_temps: Dict[str, float],
    low_temps: Dict[str, float],
    location: str,
    target_date: date,
) -> bool:
    """Detect if a frontal passage is expected.

    Frontal passages typically cause:
    - Large temperature swings
    - High model disagreement
    - Significant day-to-day changes

    Returns:
        True if frontal passage is likely
    """
    import statistics

    if len(high_temps) < 2:
        return False

    # High model disagreement suggests front
    highs = list(high_temps.values())
    lows = list(low_temps.values())

    spread_high = statistics.stdev(highs)
    spread_low = statistics.stdev(lows)

    # Large spread suggests weather transition
    if spread_high > 5.0 or spread_low > 5.0:
        return True

    # Large diurnal range also suggests frontal activity
    avg_high = statistics.mean(highs)
    avg_low = statistics.mean(lows)
    diurnal_range = avg_high - avg_low

    if diurnal_range > 35:  # > 35F swing unusual unless frontal
        return True

    return False


def _detect_extreme_temps(
    high_temps: Dict[str, float],
    low_temps: Dict[str, float],
    location: str,
    target_date: date,
) -> bool:
    """Detect if temperatures are extreme for the location/season.

    Returns:
        True if temps are unusually extreme
    """
    import statistics

    if not high_temps or not low_temps:
        return False

    avg_high = statistics.mean(list(high_temps.values()))
    avg_low = statistics.mean(list(low_temps.values()))

    # Simple heuristics for extreme temps
    # Real implementation would use historical climatology
    month = target_date.month

    # Winter months
    if month in [12, 1, 2]:
        if avg_high > 70 or avg_low < 0:
            return True
    # Summer months
    elif month in [6, 7, 8]:
        if avg_high > 105 or avg_low < 45:
            return True
    # Shoulder months
    else:
        if avg_high > 95 or avg_low < 20:
            return True

    return False


def _check_near_threshold(
    high_temps: Dict[str, float],
    low_temps: Dict[str, float],
    typical_thresholds: List[float] = None,
) -> bool:
    """Check if temperature is near common Kalshi thresholds.

    Being near a threshold increases uncertainty for binary bets.

    Returns:
        True if temperature is within 3F of a common threshold
    """
    import statistics

    if not high_temps:
        return False

    # Common Kalshi thresholds (in 5F increments typically)
    if typical_thresholds is None:
        typical_thresholds = list(range(20, 100, 5))  # 20, 25, 30, ..., 95

    avg_high = statistics.mean(list(high_temps.values()))
    avg_low = statistics.mean(list(low_temps.values()))

    # Check if within 3F of any threshold
    for threshold in typical_thresholds:
        if abs(avg_high - threshold) <= 3:
            return True
        if abs(avg_low - threshold) <= 3:
            return True

    return False


def _get_historical_accuracy(
    location: str,
    pattern_type: str,
    days: int = 30,
) -> Tuple[Optional[float], int]:
    """Get historical accuracy for similar patterns at this location.

    Returns:
        Tuple of (accuracy 0-1, sample count)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Check if forecast_log has actuals
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='forecast_log'
    """)

    if cursor.fetchone() is None:
        conn.close()
        return None, 0

    location_normalized = _normalize_location(location)
    cutoff_date = (date.today() - timedelta(days=days)).isoformat()

    # Get forecasts with actuals for this location
    cursor.execute("""
        SELECT
            ABS(predicted_high - actual_high) as high_error,
            ABS(predicted_low - actual_low) as low_error
        FROM forecast_log
        WHERE location = ?
          AND target_date >= ?
          AND actual_high IS NOT NULL
    """, (location_normalized, cutoff_date))

    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 5:
        return None, len(rows)

    # Calculate accuracy as 1 - normalized_error
    # Assuming typical error range of 0-10F
    total_error = sum(row["high_error"] + row["low_error"] for row in rows)
    avg_error = total_error / (len(rows) * 2)

    # Convert to accuracy (0-1)
    accuracy = max(0, 1 - avg_error / 10)

    return accuracy, len(rows)


def _determine_pattern_type(
    model_agreement: float,
    is_frontal: bool,
    is_extreme: bool,
    is_near_threshold: bool,
) -> str:
    """Determine the weather pattern type.

    Returns:
        "stable", "transitional", "uncertain", or "extreme"
    """
    if is_extreme:
        return "extreme"

    if is_frontal:
        return "transitional"

    if model_agreement >= 0.7 and not is_near_threshold:
        return "stable"

    if model_agreement < 0.5 or is_near_threshold:
        return "uncertain"

    return "transitional"


def _determine_confidence(
    pattern_type: str,
    model_agreement: float,
    models_used: int,
    historical_accuracy: Optional[float],
    is_near_threshold: bool,
) -> Tuple[str, float]:
    """Determine confidence level and score.

    Returns:
        Tuple of (level "HIGH"/"MEDIUM"/"LOW"/"VERY_LOW", score 0-100)
    """
    score = 0

    # Model agreement (0-40 points)
    score += model_agreement * 40

    # Number of models (0-15 points)
    if models_used >= 5:
        score += 15
    elif models_used >= 4:
        score += 12
    elif models_used >= 3:
        score += 8
    else:
        score += 4

    # Pattern type (0-20 points)
    pattern_scores = {
        "stable": 20,
        "transitional": 10,
        "uncertain": 5,
        "extreme": 8,  # Extreme but predictable
    }
    score += pattern_scores.get(pattern_type, 5)

    # Historical accuracy (0-15 points)
    if historical_accuracy is not None:
        score += historical_accuracy * 15
    else:
        score += 7  # Unknown = middle ground

    # Near threshold penalty (-10 points)
    if is_near_threshold:
        score -= 10

    # Ensure bounds
    score = max(0, min(100, score))

    # Convert to level
    if score >= 75:
        level = "HIGH"
    elif score >= 55:
        level = "MEDIUM"
    elif score >= 35:
        level = "LOW"
    else:
        level = "VERY_LOW"

    return level, score


def _determine_recommendation(
    confidence_level: str,
    pattern_type: str,
    is_frontal: bool,
    is_near_threshold: bool,
) -> Tuple[str, str]:
    """Determine betting recommendation.

    Returns:
        Tuple of (recommendation "BET"/"CAUTION"/"AVOID", explanation)
    """
    explanations = []

    # Immediate disqualifiers
    if is_frontal:
        return "AVOID", "Frontal passage expected - high uncertainty in temperature changes"

    if confidence_level == "VERY_LOW":
        return "AVOID", "Very low confidence due to model disagreement or insufficient data"

    if is_near_threshold and confidence_level != "HIGH":
        return "CAUTION", "Temperature forecast is near market threshold - small errors matter more"

    # Pattern-based recommendations
    if pattern_type == "stable" and confidence_level in ["HIGH", "MEDIUM"]:
        return "BET", "Stable pattern with good model agreement - favorable conditions for betting"

    if pattern_type == "extreme":
        return "CAUTION", "Extreme temperature event - models may underperform in unusual conditions"

    if confidence_level == "HIGH":
        return "BET", "High confidence despite transitional pattern - models agree well"

    if confidence_level == "MEDIUM":
        return "CAUTION", "Medium confidence - consider smaller bet sizes"

    return "AVOID", "Low confidence conditions - wait for better opportunity"


def classify_pattern(
    location: str,
    target_date_str: str,
    use_live: bool = True,
) -> PatternClassification:
    """Classify the weather pattern for a location and date.

    This is the main entry point for pattern analysis. It:
    1. Gets ensemble forecasts from all models
    2. Calculates model agreement metrics
    3. Identifies pattern type (stable, transitional, uncertain, extreme)
    4. Flags special situations (frontal, threshold, extreme)
    5. Looks up historical accuracy
    6. Generates betting recommendation

    Args:
        location: City name (e.g., "NYC", "New York, NY")
        target_date_str: Date string (e.g., "2026-01-08")
        use_live: Whether to fetch live data (True) or use logged forecasts

    Returns:
        PatternClassification with complete analysis

    Example:
        >>> result = classify_pattern("NYC", "2026-01-08")
        >>> print(f"Confidence: {result.confidence_level}")
        >>> print(f"Recommendation: {result.recommendation}")
    """
    # Initialize table
    init_pattern_classifications_table()

    # Parse date
    if isinstance(target_date_str, str):
        target_date = date.fromisoformat(target_date_str)
    else:
        target_date = target_date_str

    location_normalized = _normalize_location(location)

    # Get ensemble data
    if use_live:
        high_temps, low_temps, sources = _get_live_ensemble(location_normalized)
    else:
        high_temps, low_temps, sources = _get_ensemble_data(location_normalized, target_date)

    # If no data, return unknown classification
    if not high_temps:
        return PatternClassification(
            location=location_normalized,
            target_date=target_date,
            analysis_time=datetime.now(),
            model_spread_high=0.0,
            model_spread_low=0.0,
            models_used=0,
            model_agreement_score=0.0,
            pattern_type="unknown",
            confidence_level="VERY_LOW",
            confidence_score=0.0,
            is_frontal_passage=False,
            is_near_threshold=False,
            is_extreme_temp=False,
            historical_accuracy=None,
            similar_pattern_count=0,
            recommendation="AVOID",
            explanation="No forecast data available for this location/date",
        )

    # Calculate model agreement
    spread_high, spread_low, agreement = _calculate_model_agreement(high_temps, low_temps)

    # Detect special situations
    is_frontal = _detect_frontal_passage(high_temps, low_temps, location_normalized, target_date)
    is_extreme = _detect_extreme_temps(high_temps, low_temps, location_normalized, target_date)
    is_near_threshold = _check_near_threshold(high_temps, low_temps)

    # Determine pattern type
    pattern_type = _determine_pattern_type(agreement, is_frontal, is_extreme, is_near_threshold)

    # Get historical accuracy
    hist_accuracy, sample_count = _get_historical_accuracy(location_normalized, pattern_type)

    # Determine confidence
    confidence_level, confidence_score = _determine_confidence(
        pattern_type, agreement, len(sources), hist_accuracy, is_near_threshold
    )

    # Determine recommendation
    recommendation, explanation = _determine_recommendation(
        confidence_level, pattern_type, is_frontal, is_near_threshold
    )

    result = PatternClassification(
        location=location_normalized,
        target_date=target_date,
        analysis_time=datetime.now(),
        model_spread_high=round(spread_high, 2),
        model_spread_low=round(spread_low, 2),
        models_used=len(sources),
        model_agreement_score=round(agreement, 2),
        pattern_type=pattern_type,
        confidence_level=confidence_level,
        confidence_score=round(confidence_score, 1),
        is_frontal_passage=is_frontal,
        is_near_threshold=is_near_threshold,
        is_extreme_temp=is_extreme,
        historical_accuracy=round(hist_accuracy, 2) if hist_accuracy else None,
        similar_pattern_count=sample_count,
        recommendation=recommendation,
        explanation=explanation,
    )

    # Store in database
    _store_classification(result)

    return result


def _store_classification(classification: PatternClassification):
    """Store a pattern classification in the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO pattern_classifications (
            location, target_date, analysis_time,
            model_spread_high, model_spread_low, models_used, model_agreement_score,
            pattern_type, confidence_level, confidence_score,
            is_frontal_passage, is_near_threshold, is_extreme_temp,
            historical_accuracy, similar_pattern_count,
            recommendation, explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        classification.location,
        classification.target_date.isoformat(),
        classification.analysis_time.isoformat(),
        classification.model_spread_high,
        classification.model_spread_low,
        classification.models_used,
        classification.model_agreement_score,
        classification.pattern_type,
        classification.confidence_level,
        classification.confidence_score,
        1 if classification.is_frontal_passage else 0,
        1 if classification.is_near_threshold else 0,
        1 if classification.is_extreme_temp else 0,
        classification.historical_accuracy,
        classification.similar_pattern_count,
        classification.recommendation,
        classification.explanation,
    ))

    conn.commit()
    conn.close()


def get_pattern_history(
    location: str,
    days: int = 7,
) -> List[PatternClassification]:
    """Get pattern classification history for a location.

    Args:
        location: City name
        days: Number of days of history

    Returns:
        List of PatternClassification objects
    """
    init_pattern_classifications_table()

    conn = get_connection()
    cursor = conn.cursor()

    location_normalized = _normalize_location(location)
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT * FROM pattern_classifications
        WHERE location = ? AND target_date >= ?
        ORDER BY target_date DESC
    """, (location_normalized, cutoff))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append(PatternClassification(
            location=row["location"],
            target_date=date.fromisoformat(row["target_date"]),
            analysis_time=datetime.fromisoformat(row["analysis_time"]),
            model_spread_high=row["model_spread_high"],
            model_spread_low=row["model_spread_low"],
            models_used=row["models_used"],
            model_agreement_score=row["model_agreement_score"],
            pattern_type=row["pattern_type"],
            confidence_level=row["confidence_level"],
            confidence_score=row["confidence_score"],
            is_frontal_passage=bool(row["is_frontal_passage"]),
            is_near_threshold=bool(row["is_near_threshold"]),
            is_extreme_temp=bool(row["is_extreme_temp"]),
            historical_accuracy=row["historical_accuracy"],
            similar_pattern_count=row["similar_pattern_count"],
            recommendation=row["recommendation"],
            explanation=row["explanation"],
        ))

    return results


def get_pattern_report(location: str, target_date_str: str) -> str:
    """Generate a formatted pattern report.

    Args:
        location: City name
        target_date_str: Date string

    Returns:
        Formatted report string
    """
    result = classify_pattern(location, target_date_str)

    lines = [
        f"Pattern Analysis: {result.location}",
        f"Target Date: {result.target_date}",
        "=" * 50,
        "",
        f"Pattern Type: {result.pattern_type.upper()}",
        f"Confidence: {result.confidence_level} ({result.confidence_score:.0f}/100)",
        "",
        "Model Agreement:",
        f"  Sources Used: {result.models_used}",
        f"  High Temp Spread: {result.model_spread_high}F",
        f"  Low Temp Spread: {result.model_spread_low}F",
        f"  Agreement Score: {result.model_agreement_score:.0%}",
        "",
        "Situation Flags:",
        f"  Frontal Passage: {'YES' if result.is_frontal_passage else 'No'}",
        f"  Near Threshold: {'YES' if result.is_near_threshold else 'No'}",
        f"  Extreme Temps: {'YES' if result.is_extreme_temp else 'No'}",
        "",
    ]

    if result.historical_accuracy is not None:
        lines.append(f"Historical Accuracy: {result.historical_accuracy:.0%} ({result.similar_pattern_count} samples)")
    else:
        lines.append("Historical Accuracy: Insufficient data")

    lines.extend([
        "",
        "-" * 50,
        f"RECOMMENDATION: {result.recommendation}",
        f"  {result.explanation}",
        "-" * 50,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Pattern Recognition Test[/bold cyan]",
        border_style="cyan"
    ))

    # Test with a few cities
    test_cases = [
        ("NYC", "2026-01-08"),
        ("Chicago, IL", "2026-01-08"),
        ("Miami", "2026-01-08"),
    ]

    for location, date_str in test_cases:
        console.print(f"\n[bold]Analyzing {location} for {date_str}...[/bold]")

        try:
            report = get_pattern_report(location, date_str)
            console.print(Panel(report, border_style="green"))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
