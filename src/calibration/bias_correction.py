"""Bias correction module for weather forecasts.

This module calculates and applies bias corrections to weather forecasts
based on historical errors. If a model consistently predicts temperatures
too high or too low, the bias correction adjusts future forecasts accordingly.

Example:
    If GFS predicts 50F but actuals average 55F (5F cold bias):
    - bias_high = -5.0 (predicted - actual)
    - Correction: add 5F to future GFS predictions

Usage:
    from src.calibration.bias_correction import correct_forecast, update_all_bias_corrections

    # Update bias corrections daily
    update_all_bias_corrections()

    # Correct a forecast
    corrected_temp = correct_forecast("gfs", "New York, NY", 50.0)
    # Returns 55.0 if GFS has a -5F bias for NYC

Schema:
    bias_corrections: source, location, bias_high, bias_low, sample_count, updated_at
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple

from src.config import DB_PATH
from src.db.database import get_connection


@dataclass
class BiasCorrection:
    """Bias correction coefficients for a source at a location.

    Attributes:
        source: Weather model source (gfs, ecmwf, nws, icon, gem, etc.)
        location: City name (e.g., "New York, NY")
        bias_high: Average error for high temp (predicted - actual)
                   Positive = model predicts too hot, negative = too cold
        bias_low: Average error for low temp
        sample_count: Number of forecasts used to calculate bias
        updated_at: When bias was last calculated
    """
    source: str
    location: str
    bias_high: float
    bias_low: float
    sample_count: int
    updated_at: datetime


def init_bias_corrections_table() -> None:
    """Initialize the bias_corrections table for storing correction coefficients.

    Creates table if it doesn't exist (idempotent).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bias_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            location TEXT NOT NULL,
            bias_high REAL NOT NULL,
            bias_low REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(source, location)
        )
    """)

    # Index for efficient lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_bias_corrections_source_location
        ON bias_corrections(source, location)
    """)

    conn.commit()
    conn.close()


def calculate_bias(
    source: str,
    location: str,
    days: int = 14,
) -> Optional[BiasCorrection]:
    """Calculate bias for a specific source at a specific location.

    Uses the forecast_log table to compute average error over recent forecasts.

    Args:
        source: Weather model source (e.g., "gfs", "nws")
        location: Location name (e.g., "New York, NY")
        days: Rolling window in days (default 14)

    Returns:
        BiasCorrection with calculated bias, or None if insufficient data
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

    # Calculate cutoff date for rolling window
    cutoff_date = (date.today() - timedelta(days=days)).isoformat()

    # Query for forecasts with actual values
    cursor.execute("""
        SELECT
            COUNT(*) as sample_count,
            AVG(high_error) as avg_high_error,
            AVG(low_error) as avg_low_error
        FROM forecast_log
        WHERE source = ?
          AND location = ?
          AND target_date >= ?
          AND actual_high IS NOT NULL
    """, (source, location, cutoff_date))

    row = cursor.fetchone()
    conn.close()

    if row is None or row["sample_count"] == 0:
        return None

    # Require minimum samples for reliable bias estimate
    min_samples = 3
    if row["sample_count"] < min_samples:
        return None

    return BiasCorrection(
        source=source,
        location=location,
        bias_high=row["avg_high_error"] or 0.0,
        bias_low=row["avg_low_error"] or 0.0,
        sample_count=row["sample_count"],
        updated_at=datetime.now(),
    )


def update_all_bias_corrections(days: int = 14) -> int:
    """Update bias corrections for all source/location combinations.

    Reads from forecast_log table and updates bias_corrections table.

    Args:
        days: Rolling window in days (default 14)

    Returns:
        Number of bias corrections updated
    """
    init_bias_corrections_table()

    conn = get_connection()
    cursor = conn.cursor()

    # Check if forecast_log table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='forecast_log'
    """)
    if cursor.fetchone() is None:
        conn.close()
        return 0

    # Get all unique source/location combinations with actual data
    cutoff_date = (date.today() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT DISTINCT source, location
        FROM forecast_log
        WHERE target_date >= ?
          AND actual_high IS NOT NULL
    """, (cutoff_date,))

    combinations = cursor.fetchall()
    conn.close()

    updated_count = 0

    for row in combinations:
        source = row["source"]
        location = row["location"]

        bias = calculate_bias(source, location, days)
        if bias is None:
            continue

        # Store in database
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO bias_corrections
            (source, location, bias_high, bias_low, sample_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            bias.source,
            bias.location,
            bias.bias_high,
            bias.bias_low,
            bias.sample_count,
            bias.updated_at.isoformat(),
        ))

        conn.commit()
        conn.close()
        updated_count += 1

    return updated_count


def get_bias_correction(
    source: str,
    location: str,
) -> Optional[BiasCorrection]:
    """Get stored bias correction for a source at a location.

    Args:
        source: Weather model source
        location: Location name

    Returns:
        BiasCorrection if found, None otherwise
    """
    init_bias_corrections_table()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT source, location, bias_high, bias_low, sample_count, updated_at
        FROM bias_corrections
        WHERE source = ? AND location = ?
    """, (source, location))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return BiasCorrection(
        source=row["source"],
        location=row["location"],
        bias_high=row["bias_high"],
        bias_low=row["bias_low"],
        sample_count=row["sample_count"],
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def correct_forecast(
    source: str,
    location: str,
    raw_temp: float,
    temp_type: str = "high",
) -> float:
    """Apply bias correction to a raw forecast temperature.

    If the model has a known bias, the correction is subtracted from the
    raw forecast to produce a more accurate prediction.

    Example:
        If model predicts 50F but historically runs 5F cold (bias = -5):
        corrected = 50 - (-5) = 55F

    Args:
        source: Weather model source (e.g., "gfs", "nws", "model")
        location: Location name (e.g., "NYC", "New York, NY")
        raw_temp: Raw forecast temperature in Fahrenheit
        temp_type: "high" or "low" to select which bias to apply

    Returns:
        Bias-corrected temperature in Fahrenheit
    """
    # Normalize location name for lookup
    location_normalized = _normalize_location(location)

    # Look up bias correction
    bias = get_bias_correction(source, location_normalized)

    # If no bias data, return raw temperature
    if bias is None:
        # Try without normalization
        bias = get_bias_correction(source, location)

    if bias is None:
        return raw_temp

    # Apply correction: subtract bias
    # If bias is positive (model predicts too hot), subtract to cool down
    # If bias is negative (model predicts too cold), subtracting negative adds heat
    if temp_type == "high":
        return raw_temp - bias.bias_high
    else:
        return raw_temp - bias.bias_low


def _normalize_location(location: str) -> str:
    """Normalize location name for consistent lookup.

    Args:
        location: Location name in various formats

    Returns:
        Normalized location name (e.g., "New York, NY")
    """
    # Common abbreviations to full location names
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
        "AUSTIN": "Austin, TX",
        "MINNEAPOLIS": "Minneapolis, MN",
    }

    upper = location.upper().strip()
    if upper in location_map:
        return location_map[upper]

    return location


def get_all_bias_corrections() -> List[BiasCorrection]:
    """Get all stored bias corrections.

    Returns:
        List of all BiasCorrection records
    """
    init_bias_corrections_table()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT source, location, bias_high, bias_low, sample_count, updated_at
        FROM bias_corrections
        ORDER BY location, source
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        BiasCorrection(
            source=row["source"],
            location=row["location"],
            bias_high=row["bias_high"],
            bias_low=row["bias_low"],
            sample_count=row["sample_count"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
        for row in rows
    ]


def get_bias_report() -> str:
    """Generate a formatted report of all bias corrections.

    Returns:
        Formatted string report
    """
    corrections = get_all_bias_corrections()

    if not corrections:
        return "No bias corrections calculated yet. Run update_all_bias_corrections() first."

    lines = [
        "Bias Correction Report",
        "=" * 70,
        "",
        f"{'Source':<12} {'Location':<20} {'Bias High':>10} {'Bias Low':>10} {'Samples':>8}",
        "-" * 70,
    ]

    # Group by location
    by_location: Dict[str, List[BiasCorrection]] = {}
    for bc in corrections:
        if bc.location not in by_location:
            by_location[bc.location] = []
        by_location[bc.location].append(bc)

    for location in sorted(by_location.keys()):
        location_biases = by_location[location]

        for bc in sorted(location_biases, key=lambda x: x.source):
            high_sign = "+" if bc.bias_high >= 0 else ""
            low_sign = "+" if bc.bias_low >= 0 else ""

            lines.append(
                f"{bc.source:<12} {bc.location:<20} "
                f"{high_sign}{bc.bias_high:>9.2f}F {low_sign}{bc.bias_low:>9.2f}F "
                f"{bc.sample_count:>8}"
            )

    lines.append("")
    lines.append(f"Total source/location pairs: {len(corrections)}")
    lines.append("Interpretation: + bias means model predicts too hot, - means too cold")
    lines.append("Correction: Subtract bias from raw forecast for calibrated prediction")

    return "\n".join(lines)
