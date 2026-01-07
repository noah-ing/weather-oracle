"""Multi-source weather forecast API integration.

This module fetches forecasts from multiple sources:
- Open-Meteo with different models (GFS, ECMWF, ICON, GEM)
- NWS API (api.weather.gov) for official US forecasts
- WeatherAPI.com as a backup source

All sources return a standardized WeatherForecast dataclass.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import requests

from src.config import API_RATE_LIMIT_SECONDS


# Rate limiting
_last_request_time: float = 0.0


def _rate_limit() -> None:
    """Enforce rate limiting between API requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < API_RATE_LIMIT_SECONDS:
        time.sleep(API_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


@dataclass
class WeatherForecast:
    """Standardized weather forecast from any source.

    All temperature values are in Fahrenheit.
    All wind speeds are in mph.
    All precipitation values are in inches.
    """
    source: str  # e.g., "gfs", "ecmwf", "nws", "weatherapi"
    location: str  # City name or lat/lon
    lat: float
    lon: float
    forecast_time: datetime  # When forecast was made
    target_date: date  # Date being forecasted

    # Daily forecasts
    high_temp: float  # Fahrenheit
    low_temp: float  # Fahrenheit
    precip_probability: float  # 0-100%
    precip_amount: float  # inches
    wind_speed_max: float  # mph
    conditions: str  # e.g., "Partly Cloudy", "Rain"

    # Optional hourly data (for sources that provide it)
    hourly_temps: List[float] = field(default_factory=list)  # Fahrenheit
    hourly_precip_prob: List[float] = field(default_factory=list)  # 0-100%
    hourly_wind: List[float] = field(default_factory=list)  # mph
    hourly_times: List[datetime] = field(default_factory=list)

    # Confidence/quality indicators
    confidence: Optional[float] = None  # 0-1 if available
    raw_data: Optional[Dict[str, Any]] = None  # Original API response


# ============================================================================
# Open-Meteo with Multiple Models
# ============================================================================

OPEN_METEO_MODELS = {
    "gfs": "gfs_seamless",  # NOAA GFS - US-based
    "ecmwf": "ecmwf_ifs04",  # ECMWF IFS - European, very accurate
    "icon": "icon_seamless",  # DWD ICON - German, good for Europe
    "gem": "gem_seamless",  # Canadian GEM
}


def _celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def _kmh_to_mph(kmh: float) -> float:
    """Convert km/h to mph."""
    return kmh * 0.621371


def _mm_to_inches(mm: float) -> float:
    """Convert mm to inches."""
    return mm * 0.0393701


def get_open_meteo_forecast(
    lat: float,
    lon: float,
    model: str = "gfs",
    days: int = 7,
) -> Optional[WeatherForecast]:
    """
    Fetch forecast from Open-Meteo using a specific model.

    Args:
        lat: Latitude
        lon: Longitude
        model: Model name (gfs, ecmwf, icon, gem)
        days: Days ahead to forecast (1-16)

    Returns:
        WeatherForecast for tomorrow, or None if request fails
    """
    _rate_limit()

    model_param = OPEN_METEO_MODELS.get(model, model)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "models": model_param,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,weather_code",
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
        "forecast_days": min(days, 16),
        "timezone": "auto",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Get tomorrow's data (index 1)
        daily = data.get("daily", {})
        hourly = data.get("hourly", {})

        if not daily or not daily.get("time"):
            return None

        # Target tomorrow's forecast
        target_idx = 1 if len(daily["time"]) > 1 else 0
        target_date = date.fromisoformat(daily["time"][target_idx])

        # Check if this model has actual data (some models return all nulls)
        high_temp = daily.get("temperature_2m_max", [None])[target_idx]
        low_temp = daily.get("temperature_2m_min", [None])[target_idx]
        if high_temp is None or low_temp is None:
            # Model has no data for this location, skip it
            return None

        # Get hourly data for target date
        hourly_temps = []
        hourly_precip = []
        hourly_wind = []
        hourly_times = []

        if hourly and hourly.get("time"):
            for i, t in enumerate(hourly["time"]):
                dt = datetime.fromisoformat(t)
                if dt.date() == target_date:
                    hourly_times.append(dt)
                    hourly_temps.append(hourly.get("temperature_2m", [0] * len(hourly["time"]))[i] or 0)
                    hourly_precip.append(hourly.get("precipitation_probability", [0] * len(hourly["time"]))[i] or 0)
                    hourly_wind.append(hourly.get("wind_speed_10m", [0] * len(hourly["time"]))[i] or 0)

        # Map weather code to conditions
        weather_code = daily.get("weather_code", [0])[target_idx] or 0
        conditions = _weather_code_to_text(weather_code)

        return WeatherForecast(
            source=model,
            location=f"{lat:.2f},{lon:.2f}",
            lat=lat,
            lon=lon,
            forecast_time=datetime.now(),
            target_date=target_date,
            high_temp=high_temp,  # Already validated above
            low_temp=low_temp,  # Already validated above
            precip_probability=daily.get("precipitation_probability_max", [0])[target_idx] or 0,
            precip_amount=daily.get("precipitation_sum", [0])[target_idx] or 0,
            wind_speed_max=daily.get("wind_speed_10m_max", [0])[target_idx] or 0,
            conditions=conditions,
            hourly_temps=hourly_temps,
            hourly_precip_prob=hourly_precip,
            hourly_wind=hourly_wind,
            hourly_times=hourly_times,
            raw_data=data,
        )

    except Exception as e:
        print(f"Open-Meteo {model} error: {e}")
        return None


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "Clear",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing Rime Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",
        56: "Light Freezing Drizzle",
        57: "Dense Freezing Drizzle",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        66: "Light Freezing Rain",
        67: "Heavy Freezing Rain",
        71: "Slight Snow",
        73: "Moderate Snow",
        75: "Heavy Snow",
        77: "Snow Grains",
        80: "Slight Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",
        85: "Slight Snow Showers",
        86: "Heavy Snow Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with Slight Hail",
        99: "Thunderstorm with Heavy Hail",
    }
    return codes.get(code, "Unknown")


# ============================================================================
# NWS API (National Weather Service)
# ============================================================================

def get_nws_forecast(lat: float, lon: float) -> Optional[WeatherForecast]:
    """
    Fetch forecast from NWS API (api.weather.gov).

    NWS is authoritative for US weather but requires two API calls:
    1. Get grid point info from lat/lon
    2. Get forecast from grid point

    Args:
        lat: Latitude (US only)
        lon: Longitude (US only)

    Returns:
        WeatherForecast for tomorrow, or None if request fails
    """
    _rate_limit()

    headers = {
        "User-Agent": "(WeatherOracle, weather-oracle@example.com)",
        "Accept": "application/geo+json",
    }

    try:
        # Step 1: Get grid point from lat/lon
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        points_response = requests.get(points_url, headers=headers, timeout=30)
        points_response.raise_for_status()
        points_data = points_response.json()

        forecast_url = points_data["properties"]["forecast"]
        forecast_hourly_url = points_data["properties"]["forecastHourly"]

        _rate_limit()

        # Step 2: Get the forecast
        forecast_response = requests.get(forecast_url, headers=headers, timeout=30)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        # Find tomorrow's periods
        periods = forecast_data["properties"]["periods"]
        tomorrow = (datetime.now().date() + timedelta(days=1))

        # NWS gives day/night periods, find tomorrow's
        day_period = None
        night_period = None

        for period in periods:
            start_time = datetime.fromisoformat(period["startTime"].replace("Z", "+00:00"))
            if start_time.date() == tomorrow:
                if period["isDaytime"]:
                    day_period = period
                else:
                    night_period = period

            # Stop if we've gone past tomorrow
            if start_time.date() > tomorrow and day_period and night_period:
                break

        if not day_period:
            # Use first available if tomorrow not found
            day_period = periods[0]
            night_period = periods[1] if len(periods) > 1 else periods[0]
            tomorrow = datetime.fromisoformat(day_period["startTime"].replace("Z", "+00:00")).date()

        # Extract temps
        high_temp = day_period["temperature"] if day_period["isDaytime"] else night_period.get("temperature", 0)
        low_temp = night_period["temperature"] if not night_period["isDaytime"] else day_period.get("temperature", 0)

        # Ensure high > low
        if high_temp < low_temp:
            high_temp, low_temp = low_temp, high_temp

        # Parse precipitation probability from short forecast
        precip_prob = 0.0
        short_forecast = day_period.get("shortForecast", "")
        detailed = day_period.get("detailedForecast", "")

        # Look for percentage in forecast text
        import re
        prob_match = re.search(r"(\d+)\s*%\s*(?:chance|probability)", detailed, re.IGNORECASE)
        if prob_match:
            precip_prob = float(prob_match.group(1))
        elif any(word in short_forecast.lower() for word in ["rain", "showers", "thunderstorm", "snow"]):
            precip_prob = 50.0  # Default if precipitation mentioned but no %

        # Wind speed
        wind_speed = 0.0
        wind_text = day_period.get("windSpeed", "")
        wind_match = re.search(r"(\d+)", wind_text)
        if wind_match:
            wind_speed = float(wind_match.group(1))

        return WeatherForecast(
            source="nws",
            location=f"{lat:.2f},{lon:.2f}",
            lat=lat,
            lon=lon,
            forecast_time=datetime.now(),
            target_date=tomorrow,
            high_temp=float(high_temp),
            low_temp=float(low_temp),
            precip_probability=precip_prob,
            precip_amount=0.0,  # NWS doesn't give amount in this endpoint
            wind_speed_max=wind_speed,
            conditions=short_forecast,
            confidence=0.9,  # NWS is authoritative for US
            raw_data=forecast_data,
        )

    except Exception as e:
        print(f"NWS API error: {e}")
        return None


# ============================================================================
# WeatherAPI.com (Backup)
# ============================================================================

def get_weatherapi_forecast(
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
) -> Optional[WeatherForecast]:
    """
    Fetch forecast from WeatherAPI.com as a backup source.

    Requires API key from https://www.weatherapi.com (free tier available).

    Args:
        lat: Latitude
        lon: Longitude
        api_key: WeatherAPI.com API key (or set WEATHERAPI_KEY env var)

    Returns:
        WeatherForecast for tomorrow, or None if request fails
    """
    _rate_limit()

    key = api_key or os.getenv("WEATHERAPI_KEY", "")
    if not key:
        # Skip silently if no API key configured
        return None

    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": key,
        "q": f"{lat},{lon}",
        "days": 2,  # Today and tomorrow
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Get tomorrow's forecast
        forecast_days = data.get("forecast", {}).get("forecastday", [])
        if len(forecast_days) < 2:
            return None

        tomorrow_data = forecast_days[1]
        day = tomorrow_data["day"]
        target_date = date.fromisoformat(tomorrow_data["date"])

        # Get hourly data
        hourly_temps = []
        hourly_precip = []
        hourly_wind = []
        hourly_times = []

        for hour in tomorrow_data.get("hour", []):
            hourly_times.append(datetime.fromisoformat(hour["time"]))
            hourly_temps.append(hour["temp_f"])
            hourly_precip.append(float(hour.get("chance_of_rain", 0)))
            hourly_wind.append(hour["wind_mph"])

        return WeatherForecast(
            source="weatherapi",
            location=data.get("location", {}).get("name", f"{lat},{lon}"),
            lat=lat,
            lon=lon,
            forecast_time=datetime.now(),
            target_date=target_date,
            high_temp=day["maxtemp_f"],
            low_temp=day["mintemp_f"],
            precip_probability=float(day.get("daily_chance_of_rain", 0)),
            precip_amount=day.get("totalprecip_in", 0),
            wind_speed_max=day["maxwind_mph"],
            conditions=day.get("condition", {}).get("text", "Unknown"),
            hourly_temps=hourly_temps,
            hourly_precip_prob=hourly_precip,
            hourly_wind=hourly_wind,
            hourly_times=hourly_times,
            raw_data=data,
        )

    except Exception as e:
        print(f"WeatherAPI error: {e}")
        return None


# ============================================================================
# Aggregator Function
# ============================================================================

def get_all_forecasts(
    lat: float,
    lon: float,
    include_backup: bool = True,
) -> Dict[str, WeatherForecast]:
    """
    Fetch forecasts from all available sources.

    Returns a dict mapping source name to WeatherForecast.
    Failed sources are excluded from the result.

    Args:
        lat: Latitude
        lon: Longitude
        include_backup: Whether to include backup API (WeatherAPI)

    Returns:
        Dict of source_name -> WeatherForecast

    Example:
        >>> forecasts = get_all_forecasts(40.7, -74.0)
        >>> for source, fc in forecasts.items():
        ...     print(f"{source}: High {fc.high_temp}F, Low {fc.low_temp}F")
    """
    forecasts = {}

    # Open-Meteo models
    for model in ["gfs", "ecmwf", "icon", "gem"]:
        fc = get_open_meteo_forecast(lat, lon, model=model)
        if fc:
            forecasts[model] = fc

    # NWS (US only - check if coordinates are in US range)
    if -170 < lon < -60 and 20 < lat < 75:  # Rough US bounding box
        fc = get_nws_forecast(lat, lon)
        if fc:
            forecasts["nws"] = fc

    # Backup API
    if include_backup:
        fc = get_weatherapi_forecast(lat, lon)
        if fc:
            forecasts["weatherapi"] = fc

    return forecasts


def get_forecast_comparison(lat: float, lon: float) -> str:
    """
    Get a formatted comparison of forecasts from all sources.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Formatted string comparing all forecasts
    """
    forecasts = get_all_forecasts(lat, lon)

    if not forecasts:
        return "No forecasts available"

    lines = [f"Forecast Comparison for {lat:.2f}, {lon:.2f}"]
    lines.append("-" * 50)

    # Get target date from first forecast
    target_date = list(forecasts.values())[0].target_date
    lines.append(f"Target Date: {target_date}")
    lines.append("")

    # Table header
    lines.append(f"{'Source':<12} {'High':>6} {'Low':>6} {'Precip%':>8} {'Wind':>6} {'Conditions'}")
    lines.append("-" * 60)

    for source, fc in sorted(forecasts.items()):
        lines.append(
            f"{source:<12} {fc.high_temp:>5.1f}F {fc.low_temp:>5.1f}F "
            f"{fc.precip_probability:>7.0f}% {fc.wind_speed_max:>5.1f} {fc.conditions[:20]}"
        )

    # Summary stats
    lines.append("")
    highs = [fc.high_temp for fc in forecasts.values()]
    lows = [fc.low_temp for fc in forecasts.values()]

    lines.append(f"High temp range: {min(highs):.1f}F - {max(highs):.1f}F (spread: {max(highs) - min(highs):.1f}F)")
    lines.append(f"Low temp range:  {min(lows):.1f}F - {max(lows):.1f}F (spread: {max(lows) - min(lows):.1f}F)")

    return "\n".join(lines)
