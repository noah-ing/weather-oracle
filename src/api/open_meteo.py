"""Open-Meteo API client for weather data fetching."""

import time
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import requests

from src.config import (
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_HISTORICAL_URL,
    API_RATE_LIMIT_SECONDS,
)


# Rate limiting: track last request time
_last_request_time: float = 0.0


def _rate_limit() -> None:
    """Enforce rate limiting between API requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < API_RATE_LIMIT_SECONDS:
        time.sleep(API_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


@dataclass
class CurrentWeather:
    """Current weather conditions."""
    temperature: float  # Celsius
    humidity: float  # Percentage (0-100)
    wind_speed: float  # km/h
    precipitation: float  # mm
    timestamp: datetime


@dataclass
class HourlyForecast:
    """Hourly weather forecast data."""
    timestamps: List[datetime]
    temperatures: List[float]  # Celsius
    humidity: List[float]  # Percentage
    wind_speed: List[float]  # km/h
    precipitation: List[float]  # mm
    precipitation_probability: List[float]  # Percentage
    cloud_cover: List[float]  # Percentage
    pressure: List[float]  # hPa


@dataclass
class HistoricalData:
    """Historical weather observations."""
    timestamps: List[datetime]
    temperatures: List[float]  # Celsius
    humidity: List[float]  # Percentage
    wind_speed: List[float]  # km/h
    precipitation: List[float]  # mm
    pressure: List[float]  # hPa
    cloud_cover: List[float]  # Percentage


def get_current_weather(lat: float, lon: float) -> CurrentWeather:
    """
    Fetch current weather conditions for a location.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        CurrentWeather dataclass with temperature, humidity, wind, precipitation

    Raises:
        requests.RequestException: If API request fails
    """
    _rate_limit()

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }

    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    current = data["current"]

    return CurrentWeather(
        temperature=current["temperature_2m"],
        humidity=current["relative_humidity_2m"],
        wind_speed=current["wind_speed_10m"],
        precipitation=current["precipitation"],
        timestamp=datetime.fromisoformat(current["time"]),
    )


def get_forecast(lat: float, lon: float, days: int = 7) -> HourlyForecast:
    """
    Fetch hourly weather forecast for a location.

    Args:
        lat: Latitude
        lon: Longitude
        days: Number of days to forecast (1-16, default 7)

    Returns:
        HourlyForecast dataclass with hourly predictions

    Raises:
        requests.RequestException: If API request fails
        ValueError: If days is out of range
    """
    if not 1 <= days <= 16:
        raise ValueError("days must be between 1 and 16")

    _rate_limit()

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,precipitation_probability,wind_speed_10m,cloud_cover,pressure_msl",
        "forecast_days": days,
        "timezone": "auto",
    }

    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    return HourlyForecast(
        timestamps=[datetime.fromisoformat(t) for t in hourly["time"]],
        temperatures=hourly["temperature_2m"],
        humidity=hourly["relative_humidity_2m"],
        wind_speed=hourly["wind_speed_10m"],
        precipitation=hourly["precipitation"],
        precipitation_probability=hourly["precipitation_probability"],
        cloud_cover=hourly["cloud_cover"],
        pressure=hourly["pressure_msl"],
    )


def get_historical(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> HistoricalData:
    """
    Fetch historical weather observations for a location.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date for historical data
        end_date: End date for historical data

    Returns:
        HistoricalData dataclass with past observations

    Raises:
        requests.RequestException: If API request fails
        ValueError: If date range is invalid
    """
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    _rate_limit()

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl,cloud_cover",
        "timezone": "auto",
    }

    response = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    # Handle potential None values in historical data
    def clean_list(lst: List[Optional[float]]) -> List[float]:
        return [v if v is not None else 0.0 for v in lst]

    return HistoricalData(
        timestamps=[datetime.fromisoformat(t) for t in hourly["time"]],
        temperatures=clean_list(hourly["temperature_2m"]),
        humidity=clean_list(hourly["relative_humidity_2m"]),
        wind_speed=clean_list(hourly["wind_speed_10m"]),
        precipitation=clean_list(hourly["precipitation"]),
        pressure=clean_list(hourly["pressure_msl"]),
        cloud_cover=clean_list(hourly["cloud_cover"]),
    )
