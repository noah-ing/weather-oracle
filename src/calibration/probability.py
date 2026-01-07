"""Probabilistic calibration for threshold weather questions.

This module builds historical error distributions and uses them to calculate
calibrated probabilities for threshold questions like "will it be above 53°F?"

Key insight: If a model predicts 50°F but historically has ±4°F error,
then P(>53) is approximately 22%, not 0%. Raw model predictions don't
account for forecast uncertainty - this module does.

Approach:
1. Build error distribution from forecast_log (predicted - actual)
2. Model uncertainty as Gaussian with source-specific std dev
3. Widen uncertainty for longer lead times (predictions further out)
4. Use CDF to calculate probability of exceeding threshold

Usage:
    from src.calibration.probability import get_threshold_probability

    # If model predicts 50°F, what's probability temp will be > 53°F?
    prob = get_threshold_probability(50, 53, ">")
    # Returns ~0.22 if model has 4°F std dev

    # Probability temp will be < 45°F?
    prob = get_threshold_probability(50, 45, "<")
    # Returns ~0.11

For Kalshi markets:
    "NYC high above 53°F" with model prediction 50°F:
    - Naive: 0% (model says 50, so definitely not)
    - Calibrated: 22% (model has error, could be 53+)
"""

import sqlite3
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from scipy import stats
import numpy as np

from src.config import DB_PATH
from src.db.database import get_connection


@dataclass
class ErrorDistribution:
    """Historical error distribution for a source at a location.

    Attributes:
        source: Weather model source (gfs, nws, icon, etc.)
        location: City name (e.g., "New York, NY") or None for all locations
        mean_error: Average error (bias) - positive = predicts too hot
        std_error: Standard deviation of errors
        sample_count: Number of forecasts used to build distribution
        temp_type: "high" or "low" temperature errors
    """
    source: str
    location: Optional[str]
    mean_error: float  # bias
    std_error: float  # uncertainty
    sample_count: int
    temp_type: str


@dataclass
class CalibrationConfig:
    """Configuration for probabilistic calibration.

    Attributes:
        default_std: Default std dev when no data available (°F)
        min_std: Minimum std dev to prevent overconfidence (°F)
        lead_time_factor: Std dev multiplier per day of lead time
        min_samples: Minimum samples needed for reliable estimate
    """
    default_std: float = 4.0  # Typical forecast uncertainty
    min_std: float = 2.0  # Never be more confident than ±2°F
    lead_time_factor: float = 0.5  # Add 0.5°F std per day lead time
    min_samples: int = 5  # Need at least 5 forecasts


# Global default configuration
DEFAULT_CONFIG = CalibrationConfig()


def get_error_distribution(
    source: str = "all",
    location: Optional[str] = None,
    temp_type: str = "high",
    days: int = 30,
) -> Optional[ErrorDistribution]:
    """Build error distribution from historical forecast data.

    Args:
        source: Weather model source, or "all" to aggregate all sources
        location: Specific location or None for all locations
        temp_type: "high" or "low" temperature
        days: Rolling window in days

    Returns:
        ErrorDistribution with mean and std of errors, or None if insufficient data
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Check if forecast_log table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='forecast_log'
    """)
    if cursor.fetchone() is None:
        conn.close()
        return None

    # Calculate cutoff date
    cutoff_date = (date.today() - timedelta(days=days)).isoformat()

    # Select appropriate error column
    error_col = "high_error" if temp_type == "high" else "low_error"

    # Build query
    query = f"""
        SELECT {error_col} as error
        FROM forecast_log
        WHERE target_date >= ?
          AND actual_high IS NOT NULL
          AND {error_col} IS NOT NULL
    """
    params: List = [cutoff_date]

    if source != "all":
        query += " AND source = ?"
        params.append(source)

    if location:
        query += " AND location = ?"
        params.append(location)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < DEFAULT_CONFIG.min_samples:
        return None

    # Calculate statistics
    errors = [row["error"] for row in rows]
    mean_error = sum(errors) / len(errors)

    # Calculate std dev
    variance = sum((e - mean_error) ** 2 for e in errors) / len(errors)
    std_error = math.sqrt(variance) if variance > 0 else DEFAULT_CONFIG.min_std

    # Enforce minimum std dev
    std_error = max(std_error, DEFAULT_CONFIG.min_std)

    return ErrorDistribution(
        source=source,
        location=location,
        mean_error=mean_error,
        std_error=std_error,
        sample_count=len(errors),
        temp_type=temp_type,
    )


def get_adjusted_std(
    base_std: float,
    lead_days: int = 1,
) -> float:
    """Adjust standard deviation based on lead time.

    Forecast uncertainty increases with lead time. A 1-day forecast
    is more accurate than a 5-day forecast.

    Args:
        base_std: Base standard deviation from historical errors
        lead_days: Days ahead for the forecast (1 = tomorrow)

    Returns:
        Adjusted std dev accounting for lead time
    """
    # Increase uncertainty with lead time
    # Day 1: base_std
    # Day 2: base_std + 0.5°F
    # Day 5: base_std + 2.0°F
    adjustment = DEFAULT_CONFIG.lead_time_factor * (lead_days - 1)
    return base_std + adjustment


def get_threshold_probability(
    predicted_temp: float,
    threshold: float,
    comparison: str,
    source: str = "all",
    location: Optional[str] = None,
    temp_type: str = "high",
    lead_days: int = 1,
) -> float:
    """Calculate calibrated probability that actual temp will exceed/fall below threshold.

    This is the core function for probabilistic calibration. Given a model's
    point prediction, it returns the probability that the actual temperature
    will meet the threshold condition, accounting for historical forecast errors.

    Args:
        predicted_temp: Model's predicted temperature (°F)
        threshold: Threshold to compare against (°F)
        comparison: ">" (above), "<" (below), ">=" (at or above), "<=" (at or below)
        source: Weather model source, or "all" for aggregate
        location: Specific location or None for all
        temp_type: "high" or "low" temperature
        lead_days: Days ahead for forecast (1 = tomorrow)

    Returns:
        Probability (0.0 to 1.0) that actual temp meets threshold condition

    Example:
        >>> get_threshold_probability(50, 53, ">")
        0.227  # About 23% chance of exceeding 53°F

        >>> get_threshold_probability(50, 45, "<")
        0.106  # About 11% chance of being below 45°F
    """
    # Get error distribution
    dist = get_error_distribution(source, location, temp_type)

    if dist:
        # We have historical data
        bias = dist.mean_error
        std = get_adjusted_std(dist.std_error, lead_days)
    else:
        # No historical data - use defaults
        bias = 0.0
        std = get_adjusted_std(DEFAULT_CONFIG.default_std, lead_days)

    # Model prediction corrected for bias
    # If bias is +2 (model runs hot), actual is likely 2°F lower
    # So: actual_expected = predicted - bias
    expected_actual = predicted_temp - bias

    # Calculate probability using normal CDF
    # P(actual > threshold) = 1 - CDF(threshold)
    # P(actual < threshold) = CDF(threshold)

    # Normalize: z = (threshold - expected) / std
    z = (threshold - expected_actual) / std

    # Use scipy for accurate CDF calculation
    cdf_value = stats.norm.cdf(z)

    # Return appropriate probability based on comparison
    if comparison in (">", ">="):
        # P(actual > threshold) = 1 - CDF(threshold)
        # For >=, same logic since we're using continuous distribution
        prob = 1.0 - cdf_value
    elif comparison in ("<", "<="):
        # P(actual < threshold) = CDF(threshold)
        prob = cdf_value
    else:
        raise ValueError(f"Invalid comparison operator: {comparison}. Use >, <, >=, or <=")

    # Clamp to reasonable range (5% to 95%)
    # Prevents extreme overconfidence
    return max(0.05, min(0.95, prob))


def get_confidence_interval(
    predicted_temp: float,
    confidence: float = 0.95,
    source: str = "all",
    location: Optional[str] = None,
    temp_type: str = "high",
    lead_days: int = 1,
) -> Tuple[float, float]:
    """Calculate confidence interval for a temperature prediction.

    Args:
        predicted_temp: Model's predicted temperature (°F)
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        source: Weather model source
        location: Specific location
        temp_type: "high" or "low"
        lead_days: Days ahead

    Returns:
        Tuple of (lower_bound, upper_bound) in °F

    Example:
        >>> get_confidence_interval(50, 0.95)
        (42.16, 57.84)  # 95% CI with 4°F std
    """
    dist = get_error_distribution(source, location, temp_type)

    if dist:
        bias = dist.mean_error
        std = get_adjusted_std(dist.std_error, lead_days)
    else:
        bias = 0.0
        std = get_adjusted_std(DEFAULT_CONFIG.default_std, lead_days)

    expected_actual = predicted_temp - bias

    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    lower = expected_actual - z * std
    upper = expected_actual + z * std

    return (lower, upper)


def get_distribution_summary(
    source: str = "all",
    location: Optional[str] = None,
) -> Dict[str, ErrorDistribution]:
    """Get summary of error distributions for high and low temps.

    Args:
        source: Weather model source
        location: Specific location

    Returns:
        Dict with "high" and "low" ErrorDistribution objects
    """
    result = {}

    for temp_type in ["high", "low"]:
        dist = get_error_distribution(source, location, temp_type)
        if dist:
            result[temp_type] = dist

    return result


def get_calibration_report(days: int = 30) -> str:
    """Generate a formatted report of error distributions by source.

    Args:
        days: Rolling window in days

    Returns:
        Formatted string report
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Check if forecast_log table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='forecast_log'
    """)
    if cursor.fetchone() is None:
        conn.close()
        return "No forecast data available. Run log_forecasts() first."

    # Get all sources
    cutoff_date = (date.today() - timedelta(days=days)).isoformat()
    cursor.execute("""
        SELECT DISTINCT source FROM forecast_log
        WHERE target_date >= ? AND actual_high IS NOT NULL
    """, (cutoff_date,))
    sources = [row["source"] for row in cursor.fetchall()]
    conn.close()

    if not sources:
        return "No forecast accuracy data available yet."

    lines = [
        f"Probabilistic Calibration Report (Last {days} Days)",
        "=" * 70,
        "",
        "Error distributions by source (used for threshold probability calculation):",
        "",
        f"{'Source':<12} {'High Bias':>10} {'High Std':>10} {'Low Bias':>10} {'Low Std':>10} {'Samples':>8}",
        "-" * 70,
    ]

    # Add aggregate first
    dist_high = get_error_distribution("all", None, "high", days)
    dist_low = get_error_distribution("all", None, "low", days)

    if dist_high and dist_low:
        lines.append(
            f"{'ALL (agg)':<12} "
            f"{dist_high.mean_error:>+9.2f}F "
            f"{dist_high.std_error:>9.2f}F "
            f"{dist_low.mean_error:>+9.2f}F "
            f"{dist_low.std_error:>9.2f}F "
            f"{dist_high.sample_count:>8}"
        )
        lines.append("-" * 70)

    # Per-source distributions
    for source in sorted(sources):
        dist_high = get_error_distribution(source, None, "high", days)
        dist_low = get_error_distribution(source, None, "low", days)

        if dist_high and dist_low:
            lines.append(
                f"{source:<12} "
                f"{dist_high.mean_error:>+9.2f}F "
                f"{dist_high.std_error:>9.2f}F "
                f"{dist_low.mean_error:>+9.2f}F "
                f"{dist_low.std_error:>9.2f}F "
                f"{dist_high.sample_count:>8}"
            )

    lines.extend([
        "",
        "Example threshold calculations (50°F prediction, '>53°F' question):",
        "",
    ])

    # Example calculations
    for source in ["all"] + sorted(sources):
        prob = get_threshold_probability(50, 53, ">", source)
        lines.append(f"  {source}: P(>53°F) = {prob:.1%}")

    lines.extend([
        "",
        "Interpretation:",
        "  - Bias: + means model predicts too hot, subtract for correction",
        "  - Std: Spread of errors, larger = less confident predictions",
        "  - Lead time adds +0.5°F std per additional day",
    ])

    return "\n".join(lines)


def calibrate_market_probability(
    predicted_temp: float,
    threshold: float,
    comparison: str,
    sources: Optional[Dict[str, float]] = None,
    location: Optional[str] = None,
    temp_type: str = "high",
    lead_days: int = 1,
) -> Tuple[float, Dict[str, float]]:
    """Calculate ensemble-calibrated probability for a Kalshi-style market.

    This function is designed for integration with the edge calculator.
    It can use weights from multiple sources to produce a blended probability.

    Args:
        predicted_temp: Ensemble or best-estimate temperature (°F)
        threshold: Threshold from market (e.g., 53°F)
        comparison: ">" or "<"
        sources: Optional dict of {source: weight} for ensemble weighting
        location: Location for source-specific calibration
        temp_type: "high" or "low"
        lead_days: Days until target date

    Returns:
        Tuple of (probability, per_source_probs)

    Example:
        >>> calibrate_market_probability(
        ...     50.0, 53, ">",
        ...     sources={"gfs": 0.4, "nws": 0.4, "icon": 0.2}
        ... )
        (0.24, {"gfs": 0.22, "nws": 0.25, "icon": 0.23})
    """
    if sources is None:
        # Use aggregate calibration
        prob = get_threshold_probability(
            predicted_temp, threshold, comparison,
            source="all", location=location,
            temp_type=temp_type, lead_days=lead_days,
        )
        return prob, {"all": prob}

    # Calculate per-source probabilities
    per_source_probs = {}
    weighted_prob = 0.0
    total_weight = 0.0

    for source, weight in sources.items():
        prob = get_threshold_probability(
            predicted_temp, threshold, comparison,
            source=source, location=location,
            temp_type=temp_type, lead_days=lead_days,
        )
        per_source_probs[source] = prob
        weighted_prob += prob * weight
        total_weight += weight

    # Normalize if weights don't sum to 1
    if total_weight > 0:
        ensemble_prob = weighted_prob / total_weight
    else:
        ensemble_prob = get_threshold_probability(
            predicted_temp, threshold, comparison,
            source="all", location=location,
            temp_type=temp_type, lead_days=lead_days,
        )

    return ensemble_prob, per_source_probs
