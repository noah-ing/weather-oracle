"""SQLite database operations for weather storage."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import DB_PATH, DATA_DIR


def get_connection() -> sqlite3.Connection:
    """Get a database connection, creating the data directory if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database with required tables.

    Creates tables if they don't exist (idempotent).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create observations table with V3 extended columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            timestamp TEXT NOT NULL,
            temp REAL,
            humidity REAL,
            wind_speed REAL,
            wind_direction REAL,
            precipitation REAL,
            pressure_msl REAL,
            surface_pressure REAL,
            cloud_cover REAL,
            cloud_cover_low REAL,
            cloud_cover_mid REAL,
            cloud_cover_high REAL,
            dewpoint REAL,
            UNIQUE(city, timestamp)
        )
    """)

    # Add V3 columns if they don't exist (for existing databases)
    new_columns = [
        ("wind_direction", "REAL"),
        ("pressure_msl", "REAL"),
        ("surface_pressure", "REAL"),
        ("cloud_cover_low", "REAL"),
        ("cloud_cover_mid", "REAL"),
        ("cloud_cover_high", "REAL"),
        ("dewpoint", "REAL"),
    ]

    # Get existing columns
    cursor.execute("PRAGMA table_info(observations)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE observations ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

    # Migrate old 'pressure' column to 'pressure_msl' if needed
    if "pressure" in existing_cols and "pressure_msl" not in existing_cols:
        try:
            cursor.execute("UPDATE observations SET pressure_msl = pressure WHERE pressure_msl IS NULL")
        except sqlite3.OperationalError:
            pass

    # Create forecasts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            forecast_time TEXT NOT NULL,
            target_time TEXT NOT NULL,
            predicted_temp REAL,
            predicted_precip REAL,
            actual_temp REAL,
            actual_precip REAL,
            UNIQUE(city, forecast_time, target_time)
        )
    """)

    # Create indexes for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_observations_city_timestamp
        ON observations(city, timestamp)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_forecasts_city_target
        ON forecasts(city, target_time)
    """)

    conn.commit()
    conn.close()


def insert_observation(
    city: str,
    lat: float,
    lon: float,
    timestamp: datetime,
    temp: Optional[float] = None,
    humidity: Optional[float] = None,
    wind_speed: Optional[float] = None,
    wind_direction: Optional[float] = None,
    precipitation: Optional[float] = None,
    pressure_msl: Optional[float] = None,
    surface_pressure: Optional[float] = None,
    cloud_cover: Optional[float] = None,
    cloud_cover_low: Optional[float] = None,
    cloud_cover_mid: Optional[float] = None,
    cloud_cover_high: Optional[float] = None,
    dewpoint: Optional[float] = None,
) -> int:
    """Insert a weather observation into the database.

    Args:
        city: City name
        lat: Latitude
        lon: Longitude
        timestamp: Observation timestamp
        temp: Temperature in Celsius
        humidity: Relative humidity percentage
        wind_speed: Wind speed in km/h
        wind_direction: Wind direction in degrees (0-360)
        precipitation: Precipitation in mm
        pressure_msl: Mean sea level pressure in hPa
        surface_pressure: Surface pressure in hPa
        cloud_cover: Total cloud cover percentage
        cloud_cover_low: Low cloud cover percentage
        cloud_cover_mid: Mid cloud cover percentage
        cloud_cover_high: High cloud cover percentage
        dewpoint: Dewpoint temperature in Celsius

    Returns:
        The row ID of the inserted observation
    """
    conn = get_connection()
    cursor = conn.cursor()

    timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp

    cursor.execute("""
        INSERT OR REPLACE INTO observations
        (city, lat, lon, timestamp, temp, humidity, wind_speed, wind_direction,
         precipitation, pressure_msl, surface_pressure, cloud_cover,
         cloud_cover_low, cloud_cover_mid, cloud_cover_high, dewpoint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (city, lat, lon, timestamp_str, temp, humidity, wind_speed, wind_direction,
          precipitation, pressure_msl, surface_pressure, cloud_cover,
          cloud_cover_low, cloud_cover_mid, cloud_cover_high, dewpoint))

    row_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return row_id


def insert_observations_batch(observations: list[dict]) -> int:
    """Insert multiple observations in a single transaction.

    Args:
        observations: List of dicts with keys matching insert_observation params

    Returns:
        Number of rows inserted
    """
    if not observations:
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    rows = []
    for obs in observations:
        timestamp = obs.get("timestamp")
        timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        rows.append((
            obs.get("city"),
            obs.get("lat"),
            obs.get("lon"),
            timestamp_str,
            obs.get("temp"),
            obs.get("humidity"),
            obs.get("wind_speed"),
            obs.get("wind_direction"),
            obs.get("precipitation"),
            obs.get("pressure_msl"),
            obs.get("surface_pressure"),
            obs.get("cloud_cover"),
            obs.get("cloud_cover_low"),
            obs.get("cloud_cover_mid"),
            obs.get("cloud_cover_high"),
            obs.get("dewpoint"),
        ))

    cursor.executemany("""
        INSERT OR REPLACE INTO observations
        (city, lat, lon, timestamp, temp, humidity, wind_speed, wind_direction,
         precipitation, pressure_msl, surface_pressure, cloud_cover,
         cloud_cover_low, cloud_cover_mid, cloud_cover_high, dewpoint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    count = cursor.rowcount
    conn.commit()
    conn.close()

    return count


def insert_forecast(
    city: str,
    lat: float,
    lon: float,
    forecast_time: datetime,
    target_time: datetime,
    predicted_temp: Optional[float] = None,
    predicted_precip: Optional[float] = None,
    actual_temp: Optional[float] = None,
    actual_precip: Optional[float] = None,
) -> int:
    """Insert a forecast record into the database.

    Args:
        city: City name
        lat: Latitude
        lon: Longitude
        forecast_time: When the forecast was made
        target_time: The time being forecasted
        predicted_temp: Predicted temperature in Celsius
        predicted_precip: Predicted precipitation in mm
        actual_temp: Actual temperature (filled in later)
        actual_precip: Actual precipitation (filled in later)

    Returns:
        The row ID of the inserted forecast
    """
    conn = get_connection()
    cursor = conn.cursor()

    forecast_time_str = forecast_time.isoformat() if isinstance(forecast_time, datetime) else forecast_time
    target_time_str = target_time.isoformat() if isinstance(target_time, datetime) else target_time

    cursor.execute("""
        INSERT OR REPLACE INTO forecasts
        (city, lat, lon, forecast_time, target_time, predicted_temp, predicted_precip, actual_temp, actual_precip)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (city, lat, lon, forecast_time_str, target_time_str, predicted_temp, predicted_precip, actual_temp, actual_precip))

    row_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return row_id


def get_observations(
    city: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[dict]:
    """Get observations for a city within a time range.

    Args:
        city: City name
        start: Start datetime (inclusive)
        end: End datetime (inclusive)

    Returns:
        List of observation dicts
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM observations WHERE city = ?"
    params: list = [city]

    if start is not None:
        start_str = start.isoformat() if isinstance(start, datetime) else start
        query += " AND timestamp >= ?"
        params.append(start_str)

    if end is not None:
        end_str = end.isoformat() if isinstance(end, datetime) else end
        query += " AND timestamp <= ?"
        params.append(end_str)

    query += " ORDER BY timestamp"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_all_observations(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[dict]:
    """Get all observations within a time range.

    Args:
        start: Start datetime (inclusive)
        end: End datetime (inclusive)

    Returns:
        List of observation dicts
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM observations WHERE 1=1"
    params: list = []

    if start is not None:
        start_str = start.isoformat() if isinstance(start, datetime) else start
        query += " AND timestamp >= ?"
        params.append(start_str)

    if end is not None:
        end_str = end.isoformat() if isinstance(end, datetime) else end
        query += " AND timestamp <= ?"
        params.append(end_str)

    query += " ORDER BY city, timestamp"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_observation_count() -> int:
    """Get the total number of observations in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM observations")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_forecasts(
    city: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[dict]:
    """Get forecasts for a city within a target time range.

    Args:
        city: City name
        start: Start target_time (inclusive)
        end: End target_time (inclusive)

    Returns:
        List of forecast dicts
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM forecasts WHERE city = ?"
    params: list = [city]

    if start is not None:
        start_str = start.isoformat() if isinstance(start, datetime) else start
        query += " AND target_time >= ?"
        params.append(start_str)

    if end is not None:
        end_str = end.isoformat() if isinstance(end, datetime) else end
        query += " AND target_time <= ?"
        params.append(end_str)

    query += " ORDER BY target_time"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def update_forecast_actuals(
    city: str,
    target_time: datetime,
    actual_temp: float,
    actual_precip: float,
) -> int:
    """Update a forecast with actual values.

    Args:
        city: City name
        target_time: The target time of the forecast
        actual_temp: Actual temperature observed
        actual_precip: Actual precipitation observed

    Returns:
        Number of rows updated
    """
    conn = get_connection()
    cursor = conn.cursor()

    target_time_str = target_time.isoformat() if isinstance(target_time, datetime) else target_time

    cursor.execute("""
        UPDATE forecasts
        SET actual_temp = ?, actual_precip = ?
        WHERE city = ? AND target_time = ?
    """, (actual_temp, actual_precip, city, target_time_str))

    count = cursor.rowcount
    conn.commit()
    conn.close()

    return count
