"""Real-time forecast tracking and accuracy logging.

This module logs predictions from multiple weather sources (GFS, ECMWF, NWS, etc.)
and calculates rolling accuracy metrics to calibrate forecasts.

Schema:
    forecast_log: source, location, lat, lon, forecast_time, target_time,
                  predicted_high, predicted_low, actual_high, actual_low, error

Usage:
    # Log forecasts from all sources for all tracked cities
    log_forecasts()

    # Update with actual observations after 24h
    update_actuals()

    # Get rolling 7-day MAE per source per location
    accuracy = get_accuracy_by_source("NYC")
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple

from src.config import DB_PATH, DATA_DIR
from src.db.database import get_connection
from src.api.weather_models import get_all_forecasts, WeatherForecast
from src.api.geocoding import get_coordinates
from src.api.open_meteo import get_historical


# Cities to track (subset of training cities + Kalshi-relevant cities)
TRACKED_CITIES = [
    ("New York", "NY"),
    ("Chicago", "IL"),
    ("Los Angeles", "CA"),
    ("Miami", "FL"),
    ("Denver", "CO"),
    ("Phoenix", "AZ"),
    ("Seattle", "WA"),
    ("Boston", "MA"),
    ("Atlanta", "GA"),
    ("Dallas", "TX"),
    ("Philadelphia", "PA"),
    ("Austin", "TX"),
    ("Houston", "TX"),
    ("San Francisco", "CA"),
    ("Minneapolis", "MN"),
]


@dataclass
class ForecastLog:
    """A logged forecast prediction."""
    id: Optional[int]
    source: str  # gfs, ecmwf, nws, icon, gem, etc.
    location: str  # City name (e.g., "New York, NY")
    lat: float
    lon: float
    forecast_time: datetime  # When the forecast was made
    target_date: date  # The date being forecasted
    predicted_high: float  # Predicted high temp (Fahrenheit)
    predicted_low: float  # Predicted low temp (Fahrenheit)
    actual_high: Optional[float] = None  # Actual high (filled later)
    actual_low: Optional[float] = None  # Actual low (filled later)
    high_error: Optional[float] = None  # Predicted - Actual for high
    low_error: Optional[float] = None  # Predicted - Actual for low


@dataclass
class AccuracyReport:
    """Rolling accuracy metrics for a source at a location."""
    source: str
    location: str
    mae_high: float  # Mean Absolute Error for high temp
    mae_low: float  # Mean Absolute Error for low temp
    mae_combined: float  # Average of high and low MAE
    bias_high: float  # Average error (positive = predicts too hot)
    bias_low: float  # Average error for low temp
    sample_count: int  # Number of forecasts in the rolling window


def init_forecast_log_table() -> None:
    """Initialize the forecast_log table for tracking predictions.

    Creates table if it doesn't exist (idempotent).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecast_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            location TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            forecast_time TEXT NOT NULL,
            target_date TEXT NOT NULL,
            predicted_high REAL NOT NULL,
            predicted_low REAL NOT NULL,
            actual_high REAL,
            actual_low REAL,
            high_error REAL,
            low_error REAL,
            UNIQUE(source, location, forecast_time, target_date)
        )
    """)

    # Indexes for efficient queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_forecast_log_source_location
        ON forecast_log(source, location)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_forecast_log_target_date
        ON forecast_log(target_date)
    """)

    conn.commit()
    conn.close()


def log_forecasts(
    cities: Optional[List[Tuple[str, str]]] = None,
    target_date: Optional[date] = None,
) -> int:
    """Log forecasts from all sources for specified cities.

    Args:
        cities: List of (city, state) tuples, defaults to TRACKED_CITIES
        target_date: Date to forecast, defaults to tomorrow

    Returns:
        Number of forecasts logged
    """
    init_forecast_log_table()

    if cities is None:
        cities = TRACKED_CITIES

    if target_date is None:
        target_date = date.today() + timedelta(days=1)

    conn = get_connection()
    cursor = conn.cursor()

    forecast_time = datetime.now()
    logged_count = 0

    for city, state in cities:
        location = f"{city}, {state}"

        # Get coordinates
        coords = get_coordinates(city, state)
        if coords is None:
            print(f"  Could not geocode {location}, skipping")
            continue

        lat, lon = coords

        # Fetch forecasts from all sources
        forecasts = get_all_forecasts(lat, lon, include_backup=False)

        for source, fc in forecasts.items():
            # Skip if target date doesn't match (shouldn't happen but be safe)
            if fc.target_date != target_date:
                continue

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO forecast_log
                    (source, location, lat, lon, forecast_time, target_date,
                     predicted_high, predicted_low)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source,
                    location,
                    lat,
                    lon,
                    forecast_time.isoformat(),
                    target_date.isoformat(),
                    fc.high_temp,
                    fc.low_temp,
                ))
                logged_count += 1
            except Exception as e:
                print(f"  Error logging {source} for {location}: {e}")

    conn.commit()
    conn.close()

    return logged_count


def update_actuals(target_date: Optional[date] = None) -> int:
    """Update forecast_log with actual observed temperatures.

    Fetches actual high/low from Open-Meteo historical API and updates
    forecast_log entries that have passed (target_date is in the past).

    Args:
        target_date: Specific date to update, or None to update all pending

    Returns:
        Number of forecasts updated
    """
    init_forecast_log_table()

    conn = get_connection()
    cursor = conn.cursor()

    # Find forecasts that need actual values
    if target_date:
        cursor.execute("""
            SELECT DISTINCT location, lat, lon, target_date
            FROM forecast_log
            WHERE target_date = ? AND actual_high IS NULL
        """, (target_date.isoformat(),))
    else:
        # Get all pending forecasts where target_date has passed
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        cursor.execute("""
            SELECT DISTINCT location, lat, lon, target_date
            FROM forecast_log
            WHERE target_date <= ? AND actual_high IS NULL
        """, (yesterday,))

    pending = cursor.fetchall()

    updated_count = 0

    for row in pending:
        location = row["location"]
        lat = row["lat"]
        lon = row["lon"]
        td = date.fromisoformat(row["target_date"])

        # Fetch actual observations from historical API
        try:
            historical = get_historical(
                lat, lon,
                start_date=td,
                end_date=td,
            )

            if historical is None or not historical.hourly_temps:
                print(f"  No historical data for {location} on {td}")
                continue

            # Calculate actual high/low from hourly temps (convert C to F)
            temps_f = [t * 9/5 + 32 for t in historical.hourly_temps if t is not None]

            if not temps_f:
                continue

            actual_high = max(temps_f)
            actual_low = min(temps_f)

            # Update all forecasts for this location/date
            cursor.execute("""
                UPDATE forecast_log
                SET actual_high = ?,
                    actual_low = ?,
                    high_error = predicted_high - ?,
                    low_error = predicted_low - ?
                WHERE location = ? AND target_date = ? AND actual_high IS NULL
            """, (actual_high, actual_low, actual_high, actual_low, location, td.isoformat()))

            updated_count += cursor.rowcount

        except Exception as e:
            print(f"  Error fetching actuals for {location} on {td}: {e}")

    conn.commit()
    conn.close()

    return updated_count


def get_accuracy_by_source(
    location: Optional[str] = None,
    days: int = 7,
) -> Dict[str, Dict[str, AccuracyReport]]:
    """Calculate rolling accuracy (MAE) per source per location.

    Args:
        location: Specific location or None for all locations
        days: Rolling window in days (default 7)

    Returns:
        Dict mapping location -> source -> AccuracyReport
        If location specified, returns {location: {source: report}}
    """
    init_forecast_log_table()

    conn = get_connection()
    cursor = conn.cursor()

    # Calculate cutoff date for rolling window
    cutoff_date = (date.today() - timedelta(days=days)).isoformat()

    # Build query based on parameters
    query = """
        SELECT
            source,
            location,
            COUNT(*) as sample_count,
            AVG(ABS(high_error)) as mae_high,
            AVG(ABS(low_error)) as mae_low,
            AVG(high_error) as bias_high,
            AVG(low_error) as bias_low
        FROM forecast_log
        WHERE target_date >= ?
          AND actual_high IS NOT NULL
    """
    params = [cutoff_date]

    if location:
        query += " AND location = ?"
        params.append(location)

    query += " GROUP BY source, location"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Organize results
    results: Dict[str, Dict[str, AccuracyReport]] = {}

    for row in rows:
        loc = row["location"]
        src = row["source"]

        if loc not in results:
            results[loc] = {}

        mae_high = row["mae_high"] or 0.0
        mae_low = row["mae_low"] or 0.0

        results[loc][src] = AccuracyReport(
            source=src,
            location=loc,
            mae_high=mae_high,
            mae_low=mae_low,
            mae_combined=(mae_high + mae_low) / 2,
            bias_high=row["bias_high"] or 0.0,
            bias_low=row["bias_low"] or 0.0,
            sample_count=row["sample_count"],
        )

    return results


def get_accuracy_report(days: int = 7) -> str:
    """Generate a formatted accuracy report for all sources.

    Args:
        days: Rolling window in days

    Returns:
        Formatted string report
    """
    accuracy = get_accuracy_by_source(days=days)

    if not accuracy:
        return "No accuracy data available. Run log_forecasts() and update_actuals() first."

    lines = [
        f"Forecast Accuracy Report (Last {days} Days)",
        "=" * 60,
        "",
    ]

    # Aggregate by source across all locations
    source_totals: Dict[str, List[AccuracyReport]] = {}
    for location, sources in accuracy.items():
        for source, report in sources.items():
            if source not in source_totals:
                source_totals[source] = []
            source_totals[source].append(report)

    # Header for source summary
    lines.append(f"{'Source':<12} {'MAE High':>10} {'MAE Low':>10} {'MAE Avg':>10} {'Bias High':>10} {'Samples':>8}")
    lines.append("-" * 60)

    # Calculate and display source averages
    source_averages: List[Tuple[str, float, float, float, float, int]] = []

    for source, reports in sorted(source_totals.items()):
        total_samples = sum(r.sample_count for r in reports)
        if total_samples == 0:
            continue

        # Weighted average by sample count
        avg_mae_high = sum(r.mae_high * r.sample_count for r in reports) / total_samples
        avg_mae_low = sum(r.mae_low * r.sample_count for r in reports) / total_samples
        avg_bias_high = sum(r.bias_high * r.sample_count for r in reports) / total_samples
        avg_combined = (avg_mae_high + avg_mae_low) / 2

        source_averages.append((source, avg_mae_high, avg_mae_low, avg_combined, avg_bias_high, total_samples))

    # Sort by MAE combined (best first)
    source_averages.sort(key=lambda x: x[3])

    for source, mae_high, mae_low, mae_avg, bias_high, samples in source_averages:
        bias_sign = "+" if bias_high >= 0 else ""
        lines.append(
            f"{source:<12} {mae_high:>9.2f}F {mae_low:>9.2f}F {mae_avg:>9.2f}F "
            f"{bias_sign}{bias_high:>9.2f}F {samples:>8}"
        )

    lines.append("")
    lines.append(f"Total locations tracked: {len(accuracy)}")
    lines.append(f"Bias interpretation: + means model predicts too hot, - means too cold")

    return "\n".join(lines)


def get_forecast_count() -> int:
    """Get total number of logged forecasts."""
    init_forecast_log_table()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM forecast_log")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_pending_actuals_count() -> int:
    """Get count of forecasts waiting for actual values."""
    init_forecast_log_table()
    conn = get_connection()
    cursor = conn.cursor()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    cursor.execute("""
        SELECT COUNT(*) FROM forecast_log
        WHERE target_date <= ? AND actual_high IS NULL
    """, (yesterday,))
    count = cursor.fetchone()[0]
    conn.close()
    return count
