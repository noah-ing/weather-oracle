"""Ensemble predictions combining neural network with Open-Meteo forecasts.

Blends NN predictions with raw API forecasts using configurable weights
that can adjust based on historical accuracy per location.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

from src.api.open_meteo import get_forecast, HourlyForecast
from src.api.geocoding import get_coordinates
from src.config import DATA_DIR, SEQUENCE_OUTPUT_HOURS
from src.db.database import get_connection, init_db
from src.inference.predictor import WeatherPredictor, Forecast, HourlyPrediction


@dataclass
class EnsemblePrediction:
    """Single hour ensemble prediction with source weights."""
    timestamp: datetime
    temperature: float  # Fahrenheit
    temperature_nn: float  # NN prediction
    temperature_api: float  # API prediction
    nn_weight: float  # Weight used for NN (0-1)
    api_weight: float  # Weight used for API (0-1)
    precip_probability: float  # 0-100%
    wind_speed: float  # mph
    conditions: str


@dataclass
class EnsembleForecast:
    """24-hour ensemble forecast with source tracking."""
    location: str
    latitude: float
    longitude: float
    generated_at: datetime
    nn_weight_used: float  # Overall NN weight for this forecast
    api_weight_used: float  # Overall API weight for this forecast
    hourly: List[EnsemblePrediction]
    accuracy_history: Optional[Dict[str, float]] = None


def _celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def _kmh_to_mph(kmh: float) -> float:
    """Convert km/h to mph."""
    return kmh * 0.621371


def _get_conditions(temp_f: float, precip_prob: float, wind_mph: float) -> str:
    """Generate weather conditions text from predictions."""
    conditions = []

    if temp_f < 32:
        conditions.append("Freezing")
    elif temp_f < 50:
        conditions.append("Cold")
    elif temp_f < 68:
        conditions.append("Cool")
    elif temp_f < 80:
        conditions.append("Warm")
    else:
        conditions.append("Hot")

    if precip_prob > 70:
        conditions.append("Likely rain")
    elif precip_prob > 40:
        conditions.append("Chance of rain")
    elif precip_prob > 20:
        conditions.append("Slight chance of rain")
    else:
        conditions.append("Dry")

    if wind_mph > 25:
        conditions.append("Windy")
    elif wind_mph > 15:
        conditions.append("Breezy")

    return ", ".join(conditions)


class EnsemblePredictor:
    """Ensemble weather predictor combining NN and API forecasts.
    
    Blends predictions from a trained neural network with raw forecasts
    from Open-Meteo API. Weights can be adjusted based on historical
    accuracy for each location.
    
    Default weights: NN 60%, API 40%
    
    Example:
        >>> predictor = EnsemblePredictor()
        >>> forecast = predictor.get_ensemble_forecast(41.85, -87.65)
        >>> print(forecast.hourly[0].temperature)
    """
    
    def __init__(
        self,
        nn_weight: float = 0.6,
        api_weight: float = 0.4,
        use_adaptive_weights: bool = True,
    ):
        """Initialize ensemble predictor.
        
        Args:
            nn_weight: Default weight for NN predictions (0-1)
            api_weight: Default weight for API predictions (0-1)
            use_adaptive_weights: Whether to adjust weights based on historical accuracy
        """
        if not np.isclose(nn_weight + api_weight, 1.0):
            raise ValueError("nn_weight + api_weight must equal 1.0")
        
        self.default_nn_weight = nn_weight
        self.default_api_weight = api_weight
        self.use_adaptive_weights = use_adaptive_weights
        
        # Initialize NN predictor
        self.nn_predictor = WeatherPredictor()
        
        # Initialize accuracy tracking database
        self._init_accuracy_db()
    
    def _init_accuracy_db(self) -> None:
        """Initialize SQLite table for tracking prediction accuracy."""
        init_db()  # Ensure main db exists
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                prediction_time TEXT NOT NULL,
                target_time TEXT NOT NULL,
                nn_predicted_temp REAL,
                api_predicted_temp REAL,
                actual_temp REAL,
                nn_error REAL,
                api_error REAL,
                UNIQUE(location, prediction_time, target_time)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ensemble_accuracy_location
            ON ensemble_accuracy(location)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_location_weights(self, lat: float, lon: float) -> tuple[float, float]:
        """Get adaptive weights based on historical accuracy for a location.
        
        If no history exists or adaptive weights are disabled, returns default weights.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Tuple of (nn_weight, api_weight)
        """
        if not self.use_adaptive_weights:
            return self.default_nn_weight, self.default_api_weight
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find nearby location (within 0.5 degrees)
        cursor.execute("""
            SELECT 
                AVG(ABS(nn_error)) as avg_nn_error,
                AVG(ABS(api_error)) as avg_api_error,
                COUNT(*) as count
            FROM ensemble_accuracy
            WHERE ABS(lat - ?) < 0.5 AND ABS(lon - ?) < 0.5
            AND actual_temp IS NOT NULL
        """, (lat, lon))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None or row[2] < 10:  # Need at least 10 samples
            return self.default_nn_weight, self.default_api_weight
        
        avg_nn_error = row[0]
        avg_api_error = row[1]
        
        if avg_nn_error is None or avg_api_error is None:
            return self.default_nn_weight, self.default_api_weight
        
        # Calculate weights inversely proportional to error
        # Lower error = higher weight
        total_error = avg_nn_error + avg_api_error
        if total_error < 0.01:  # Avoid division by zero
            return self.default_nn_weight, self.default_api_weight
        
        nn_weight = avg_api_error / total_error  # Higher API error = higher NN weight
        api_weight = avg_nn_error / total_error
        
        # Clamp weights to reasonable range (0.3 - 0.7)
        nn_weight = max(0.3, min(0.7, nn_weight))
        api_weight = 1.0 - nn_weight
        
        return nn_weight, api_weight
    
    def _get_api_forecast(self, lat: float, lon: float) -> Optional[HourlyForecast]:
        """Get forecast from Open-Meteo API.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            HourlyForecast or None if request fails
        """
        try:
            return get_forecast(lat, lon, days=1)
        except Exception:
            return None
    
    def get_ensemble_forecast(
        self,
        lat: float,
        lon: float,
        location_name: Optional[str] = None,
    ) -> Optional[EnsembleForecast]:
        """Get blended ensemble forecast for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            location_name: Optional name for the location
        
        Returns:
            EnsembleForecast with blended predictions, or None if both sources fail
        """
        # Get NN prediction
        nn_forecast = self.nn_predictor.predict(lat, lon)
        
        # Get API prediction
        api_forecast = self._get_api_forecast(lat, lon)
        
        if nn_forecast is None and api_forecast is None:
            return None
        
        # Get adaptive weights
        nn_weight, api_weight = self._get_location_weights(lat, lon)
        
        # If one source fails, use 100% of the other
        if nn_forecast is None:
            nn_weight, api_weight = 0.0, 1.0
        elif api_forecast is None:
            nn_weight, api_weight = 1.0, 0.0
        
        # Generate timestamps
        now = datetime.now()
        start_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        # Blend predictions
        hourly = []
        
        for hour in range(SEQUENCE_OUTPUT_HOURS):
            timestamp = start_time + timedelta(hours=hour)
            
            # Get NN values (if available)
            if nn_forecast is not None and hour < len(nn_forecast.hourly):
                nn_temp = nn_forecast.hourly[hour].temperature
                nn_precip = nn_forecast.hourly[hour].precip_probability
                nn_wind = nn_forecast.hourly[hour].wind_speed
            else:
                nn_temp = nn_precip = nn_wind = None
            
            # Get API values (if available)
            if api_forecast is not None and hour < len(api_forecast.timestamps):
                api_temp = _celsius_to_fahrenheit(api_forecast.temperatures[hour])
                api_precip = api_forecast.precipitation_probability[hour]
                api_wind = _kmh_to_mph(api_forecast.wind_speed[hour])
            else:
                api_temp = api_precip = api_wind = None
            
            # Blend values
            if nn_temp is not None and api_temp is not None:
                blended_temp = nn_weight * nn_temp + api_weight * api_temp
                blended_precip = nn_weight * nn_precip + api_weight * api_precip
                blended_wind = nn_weight * nn_wind + api_weight * api_wind
            elif nn_temp is not None:
                blended_temp, blended_precip, blended_wind = nn_temp, nn_precip, nn_wind
                api_temp = api_precip = api_wind = 0.0
            elif api_temp is not None:
                blended_temp, blended_precip, blended_wind = api_temp, api_precip, api_wind
                nn_temp = nn_precip = nn_wind = 0.0
            else:
                continue  # Skip if no data
            
            conditions = _get_conditions(blended_temp, blended_precip, blended_wind)
            
            hourly.append(EnsemblePrediction(
                timestamp=timestamp,
                temperature=round(blended_temp, 1),
                temperature_nn=round(nn_temp, 1) if nn_temp else 0.0,
                temperature_api=round(api_temp, 1) if api_temp else 0.0,
                nn_weight=nn_weight,
                api_weight=api_weight,
                precip_probability=round(blended_precip, 1),
                wind_speed=round(blended_wind, 1),
                conditions=conditions,
            ))
        
        if not hourly:
            return None
        
        # Get accuracy history for this location
        accuracy_history = self._get_accuracy_history(lat, lon)
        
        return EnsembleForecast(
            location=location_name or f"({lat:.4f}, {lon:.4f})",
            latitude=lat,
            longitude=lon,
            generated_at=now,
            nn_weight_used=nn_weight,
            api_weight_used=api_weight,
            hourly=hourly,
            accuracy_history=accuracy_history,
        )
    
    def get_ensemble_forecast_city(
        self,
        city: str,
        state: Optional[str] = None,
    ) -> Optional[EnsembleForecast]:
        """Get ensemble forecast for a US city.
        
        Args:
            city: City name
            state: Optional state abbreviation
        
        Returns:
            EnsembleForecast or None if city not found
        """
        coords = get_coordinates(city, state)
        if coords is None:
            return None
        
        lat, lon = coords
        location_name = f"{city}, {state}" if state else city
        
        return self.get_ensemble_forecast(lat, lon, location_name)
    
    def record_prediction(
        self,
        lat: float,
        lon: float,
        location_name: str,
        prediction_time: datetime,
        target_time: datetime,
        nn_predicted_temp: float,
        api_predicted_temp: float,
    ) -> None:
        """Record a prediction for later accuracy tracking.
        
        Args:
            lat: Latitude
            lon: Longitude
            location_name: Location name
            prediction_time: When the prediction was made
            target_time: Time being predicted
            nn_predicted_temp: NN temperature prediction
            api_predicted_temp: API temperature prediction
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO ensemble_accuracy
            (location, lat, lon, prediction_time, target_time, nn_predicted_temp, api_predicted_temp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            location_name, lat, lon,
            prediction_time.isoformat(),
            target_time.isoformat(),
            nn_predicted_temp,
            api_predicted_temp,
        ))
        
        conn.commit()
        conn.close()
    
    def update_actual_values(
        self,
        location_name: str,
        target_time: datetime,
        actual_temp: float,
    ) -> None:
        """Update a recorded prediction with actual values and compute errors.
        
        Args:
            location_name: Location name
            target_time: Time that was predicted
            actual_temp: Actual observed temperature
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get the prediction
        cursor.execute("""
            SELECT nn_predicted_temp, api_predicted_temp
            FROM ensemble_accuracy
            WHERE location = ? AND target_time = ?
        """, (location_name, target_time.isoformat()))
        
        row = cursor.fetchone()
        if row is None:
            conn.close()
            return
        
        nn_pred, api_pred = row
        nn_error = abs(actual_temp - nn_pred) if nn_pred else None
        api_error = abs(actual_temp - api_pred) if api_pred else None
        
        cursor.execute("""
            UPDATE ensemble_accuracy
            SET actual_temp = ?, nn_error = ?, api_error = ?
            WHERE location = ? AND target_time = ?
        """, (actual_temp, nn_error, api_error, location_name, target_time.isoformat()))
        
        conn.commit()
        conn.close()
    
    def _get_accuracy_history(self, lat: float, lon: float) -> Optional[Dict[str, float]]:
        """Get accuracy statistics for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with accuracy statistics, or None if no history
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(ABS(nn_error)) as avg_nn_mae,
                AVG(ABS(api_error)) as avg_api_mae,
                COUNT(*) as count
            FROM ensemble_accuracy
            WHERE ABS(lat - ?) < 0.5 AND ABS(lon - ?) < 0.5
            AND actual_temp IS NOT NULL
        """, (lat, lon))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None or row[2] < 5:
            return None
        
        return {
            "nn_mae": round(row[0], 2) if row[0] else None,
            "api_mae": round(row[1], 2) if row[1] else None,
            "sample_count": row[2],
        }


def get_ensemble_forecast(lat: float, lon: float) -> Optional[EnsembleForecast]:
    """Convenience function to get ensemble forecast.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        EnsembleForecast with blended predictions
    """
    predictor = EnsemblePredictor()
    return predictor.get_ensemble_forecast(lat, lon)


def format_ensemble_forecast(forecast: EnsembleForecast) -> dict:
    """Convert an EnsembleForecast to a dictionary for JSON serialization.
    
    Args:
        forecast: EnsembleForecast object
    
    Returns:
        Dictionary representation
    """
    return {
        "location": forecast.location,
        "latitude": forecast.latitude,
        "longitude": forecast.longitude,
        "generated_at": forecast.generated_at.isoformat(),
        "nn_weight": forecast.nn_weight_used,
        "api_weight": forecast.api_weight_used,
        "accuracy_history": forecast.accuracy_history,
        "hourly": [
            {
                "timestamp": h.timestamp.isoformat(),
                "temperature": h.temperature,
                "temperature_nn": h.temperature_nn,
                "temperature_api": h.temperature_api,
                "nn_weight": h.nn_weight,
                "api_weight": h.api_weight,
                "precip_probability": h.precip_probability,
                "wind_speed": h.wind_speed,
                "conditions": h.conditions,
            }
            for h in forecast.hourly
        ],
    }


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print("\n[bold cyan]Testing Ensemble Predictor[/bold cyan]\n")
    
    try:
        predictor = EnsemblePredictor()
        console.print("Predictor initialized")
        
        # Test with Chicago
        console.print("\nFetching ensemble forecast for Chicago, IL...")
        forecast = predictor.get_ensemble_forecast_city("Chicago", "IL")
        
        if forecast:
            console.print(f"\nForecast for {forecast.location}")
            console.print(f"NN Weight: {forecast.nn_weight_used:.0%}")
            console.print(f"API Weight: {forecast.api_weight_used:.0%}")
            
            table = Table(title="Ensemble Forecast (First 6 Hours)")
            table.add_column("Time")
            table.add_column("Temp (°F)")
            table.add_column("NN")
            table.add_column("API")
            table.add_column("Precip %")
            table.add_column("Conditions")
            
            for h in forecast.hourly[:6]:
                table.add_row(
                    h.timestamp.strftime("%H:%M"),
                    f"{h.temperature:.1f}",
                    f"{h.temperature_nn:.1f}",
                    f"{h.temperature_api:.1f}",
                    f"{h.precip_probability:.0f}%",
                    h.conditions,
                )
            
            console.print(table)
            console.print("\n[green]Test passed![/green]")
        else:
            console.print("[red]Failed to get forecast[/red]")
            
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Train a model first: python -m src.training.trainer[/yellow]")
