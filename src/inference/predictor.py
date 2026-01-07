"""Inference module for making weather predictions.

Provides the WeatherPredictor class for loading a trained model and
making 24-hour weather forecasts for any location.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.api.geocoding import get_coordinates
from src.api.open_meteo import get_forecast, get_historical
from src.config import (
    DATA_DIR,
    MODELS_DIR,
    SEQUENCE_INPUT_HOURS,
    SEQUENCE_OUTPUT_HOURS,
)
from src.data.preprocessing import load_scaler, DB_INPUT_FEATURES, DB_OUTPUT_FEATURES
from src.model.weather_net import WeatherNet


@dataclass
class HourlyPrediction:
    """Single hour prediction with confidence intervals."""
    timestamp: datetime
    temperature: float  # Fahrenheit
    temperature_low: float  # Lower confidence bound
    temperature_high: float  # Upper confidence bound
    precip_probability: float  # 0-100%
    wind_speed: float  # mph
    wind_speed_low: float  # Lower confidence bound
    wind_speed_high: float  # Upper confidence bound
    conditions: str  # Text description


@dataclass
class Forecast:
    """24-hour forecast result."""
    location: str
    latitude: float
    longitude: float
    generated_at: datetime
    hourly: list[HourlyPrediction]


def _celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def _kmh_to_mph(kmh: float) -> float:
    """Convert km/h to mph."""
    return kmh * 0.621371


def _get_conditions(temp_f: float, precip_prob: float, wind_mph: float) -> str:
    """Generate weather conditions text from predictions."""
    conditions = []

    # Temperature-based conditions
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

    # Precipitation conditions
    if precip_prob > 70:
        conditions.append("Likely rain")
    elif precip_prob > 40:
        conditions.append("Chance of rain")
    elif precip_prob > 20:
        conditions.append("Slight chance of rain")
    else:
        conditions.append("Dry")

    # Wind conditions
    if wind_mph > 25:
        conditions.append("Windy")
    elif wind_mph > 15:
        conditions.append("Breezy")

    return ", ".join(conditions)


class WeatherPredictor:
    """Weather prediction using trained neural network model.

    Loads a pre-trained WeatherNet model and scaler to make
    24-hour weather forecasts for any location.

    Example:
        >>> predictor = WeatherPredictor()
        >>> forecast = predictor.predict_city("Chicago", "IL")
        >>> print(forecast.hourly[0].temperature)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        n_samples: int = 10,  # For Monte Carlo dropout uncertainty
    ):
        """Initialize the predictor with model and scaler.

        Args:
            model_path: Path to saved model checkpoint (default: models/best_model.pt)
            scaler_path: Path to saved scaler (default: data/scaler.pkl)
            n_samples: Number of samples for uncertainty estimation
        """
        self.model_path = model_path or MODELS_DIR / "best_model.pt"
        self.scaler_path = scaler_path or DATA_DIR / "scaler.pkl"
        self.n_samples = n_samples

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self._load_model()

        # Load scaler
        self._load_scaler()

    def _load_model(self) -> None:
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Create model with same architecture
        self.model = WeatherNet()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _load_scaler(self) -> None:
        """Load saved scaler for data normalization."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

        scaler_info = load_scaler(self.scaler_path)
        self.input_scaler = scaler_info["input_scaler"]
        self.output_scaler = scaler_info["output_scaler"]

    def _fetch_recent_data(self, lat: float, lon: float) -> Optional[np.ndarray]:
        """Fetch the most recent 72 hours of weather data for a location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            numpy array of shape (72, 6) with input features, or None if insufficient data
        """
        # Get historical data for the past 3-4 days to ensure we have 72 hours
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=4)

        try:
            historical = get_historical(lat, lon, start_date, end_date)
        except Exception:
            return None

        # Extract the features we need in the correct order
        # DB_INPUT_FEATURES: temp, humidity, wind_speed, precipitation, pressure, cloud_cover
        temps = historical.temperatures[-SEQUENCE_INPUT_HOURS:]
        humidity = historical.humidity[-SEQUENCE_INPUT_HOURS:]
        wind_speed = historical.wind_speed[-SEQUENCE_INPUT_HOURS:]
        precipitation = historical.precipitation[-SEQUENCE_INPUT_HOURS:]
        pressure = historical.pressure[-SEQUENCE_INPUT_HOURS:]
        cloud_cover = historical.cloud_cover[-SEQUENCE_INPUT_HOURS:]

        # Check if we have enough data
        if len(temps) < SEQUENCE_INPUT_HOURS:
            return None

        # Stack into array (72, 6)
        data = np.column_stack([
            temps,
            humidity,
            wind_speed,
            precipitation,
            pressure,
            cloud_cover,
        ])

        return data

    def _predict_with_uncertainty(self, input_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Make prediction with uncertainty estimation using Monte Carlo dropout.

        Args:
            input_data: Input data of shape (72, 6) in original scale

        Returns:
            Tuple of (mean_predictions, std_predictions) each of shape (24, 3)
        """
        # Normalize input
        input_normalized = self.input_scaler.transform(input_data)

        # Convert to tensor and add batch dimension
        x = torch.tensor(input_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Collect multiple predictions using dropout for uncertainty
        predictions = []

        # Enable dropout for Monte Carlo sampling
        self.model.train()  # Enable dropout

        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(x)
                predictions.append(output.cpu().numpy())

        # Set back to eval mode
        self.model.eval()

        # Stack predictions: (n_samples, 1, 24, 3) -> (n_samples, 24, 3)
        predictions = np.concatenate(predictions, axis=0)

        # Calculate mean and std
        mean_pred = predictions.mean(axis=0)  # (24, 3)
        std_pred = predictions.std(axis=0)  # (24, 3)

        # Inverse transform to original scale
        mean_pred_original = self.output_scaler.inverse_transform(mean_pred)

        # For std, scale by the data range
        data_range = self.output_scaler.data_range_
        std_pred_original = std_pred * data_range

        return mean_pred_original, std_pred_original

    def predict(self, lat: float, lon: float) -> Optional[Forecast]:
        """Make a 24-hour weather forecast for a location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Forecast object with hourly predictions, or None if prediction fails
        """
        # Fetch recent weather data
        input_data = self._fetch_recent_data(lat, lon)
        if input_data is None:
            return None

        # Make prediction with uncertainty
        mean_pred, std_pred = self._predict_with_uncertainty(input_data)

        # Generate timestamps for predictions (next 24 hours)
        now = datetime.now()
        start_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        # Build hourly predictions
        hourly = []
        for hour in range(SEQUENCE_OUTPUT_HOURS):
            timestamp = start_time + timedelta(hours=hour)

            # Extract values (output features: temp, precipitation, wind_speed)
            temp_c = mean_pred[hour, 0]
            precip = mean_pred[hour, 1]
            wind_kmh = mean_pred[hour, 2]

            # Get uncertainties
            temp_std = std_pred[hour, 0]
            wind_std = std_pred[hour, 2]

            # Convert units
            temp_f = _celsius_to_fahrenheit(temp_c)
            temp_low_f = _celsius_to_fahrenheit(temp_c - 1.96 * temp_std)  # 95% CI
            temp_high_f = _celsius_to_fahrenheit(temp_c + 1.96 * temp_std)

            wind_mph = _kmh_to_mph(wind_kmh)
            wind_low_mph = max(0, _kmh_to_mph(wind_kmh - 1.96 * wind_std))
            wind_high_mph = _kmh_to_mph(wind_kmh + 1.96 * wind_std)

            # Convert precipitation amount to probability (0-100%)
            # Using sigmoid-like transformation for precipitation
            precip_prob = min(100, max(0, precip * 100))  # Rough approximation

            # Get conditions text
            conditions = _get_conditions(temp_f, precip_prob, wind_mph)

            hourly.append(HourlyPrediction(
                timestamp=timestamp,
                temperature=round(temp_f, 1),
                temperature_low=round(temp_low_f, 1),
                temperature_high=round(temp_high_f, 1),
                precip_probability=round(precip_prob, 1),
                wind_speed=round(wind_mph, 1),
                wind_speed_low=round(wind_low_mph, 1),
                wind_speed_high=round(wind_high_mph, 1),
                conditions=conditions,
            ))

        return Forecast(
            location=f"({lat:.4f}, {lon:.4f})",
            latitude=lat,
            longitude=lon,
            generated_at=now,
            hourly=hourly,
        )

    def predict_city(self, city: str, state: Optional[str] = None) -> Optional[Forecast]:
        """Make a 24-hour weather forecast for a US city.

        Args:
            city: City name (e.g., "Chicago", "New York")
            state: Optional state abbreviation or name (e.g., "IL", "New York")

        Returns:
            Forecast object with hourly predictions, or None if city not found
        """
        # Resolve city to coordinates
        coords = get_coordinates(city, state)
        if coords is None:
            return None

        lat, lon = coords

        # Make prediction
        forecast = self.predict(lat, lon)

        if forecast is not None:
            # Update location name with city
            if state:
                forecast.location = f"{city}, {state}"
            else:
                forecast.location = city

        return forecast


def format_forecast(forecast: Forecast) -> dict:
    """Convert a Forecast to a dictionary for JSON serialization.

    Args:
        forecast: Forecast object

    Returns:
        Dictionary representation of the forecast
    """
    return {
        "location": forecast.location,
        "latitude": forecast.latitude,
        "longitude": forecast.longitude,
        "generated_at": forecast.generated_at.isoformat(),
        "hourly": [
            {
                "timestamp": h.timestamp.isoformat(),
                "temperature": h.temperature,
                "temperature_low": h.temperature_low,
                "temperature_high": h.temperature_high,
                "precip_probability": h.precip_probability,
                "wind_speed": h.wind_speed,
                "wind_speed_low": h.wind_speed_low,
                "wind_speed_high": h.wind_speed_high,
                "conditions": h.conditions,
            }
            for h in forecast.hourly
        ],
    }


if __name__ == "__main__":
    # Test the predictor
    print("Testing WeatherPredictor...")

    try:
        predictor = WeatherPredictor()
        print("Model and scaler loaded successfully")

        # Test with Chicago
        print("\nFetching forecast for Chicago, IL...")
        forecast = predictor.predict_city("Chicago", "IL")

        if forecast:
            print(f"\nForecast for {forecast.location}")
            print(f"Generated at: {forecast.generated_at}")
            print(f"Coordinates: ({forecast.latitude}, {forecast.longitude})")
            print("\nFirst 6 hours:")
            print("-" * 80)
            for h in forecast.hourly[:6]:
                print(f"{h.timestamp.strftime('%H:%M')}: "
                      f"{h.temperature:.1f}°F ({h.temperature_low:.1f}-{h.temperature_high:.1f}), "
                      f"Precip: {h.precip_probability:.0f}%, "
                      f"Wind: {h.wind_speed:.1f} mph, "
                      f"{h.conditions}")
        else:
            print("Failed to get forecast")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to train the model first with: python -m src.training.trainer")
