"""Regional ensemble predictor for Weather Oracle V3.

Uses region-specific transformer models trained on local weather patterns.
Falls back to nearest region for cities not in the training set.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import math

import numpy as np
import torch

from src.api.geocoding import get_coordinates
from src.api.open_meteo import get_historical
from src.config import DATA_DIR, MODELS_DIR, SEQUENCE_INPUT_HOURS, SEQUENCE_OUTPUT_HOURS
from src.data.regions import (
    get_region,
    get_all_regions,
    get_cities_in_region,
    REGIONS,
)
from src.data.preprocessing import (
    INPUT_FEATURE_NAMES,
    DB_OUTPUT_FEATURES,
    load_scaler_v3,
    encode_wind_direction,
    calculate_temp_dewpoint_spread,
    calculate_pressure_tendency,
)
from src.model.weather_transformer import WeatherTransformer


@dataclass
class RegionalPrediction:
    """Prediction from a regional model."""
    location: str
    region: str
    lat: float
    lon: float
    generated_at: datetime
    # 24-hour predictions
    temperatures: List[float]  # Fahrenheit
    precip_probabilities: List[float]  # 0-100%
    wind_speeds: List[float]  # mph
    # Aggregated values
    high_temp: float
    low_temp: float
    max_wind: float
    avg_precip_prob: float
    # Model info
    model_path: str
    confidence: float


def _celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def _kmh_to_mph(kmh: float) -> float:
    """Convert km/h to mph."""
    return kmh * 0.621371


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in km using Haversine formula."""
    R = 6371  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Region center coordinates for fallback matching
REGION_CENTERS: Dict[str, Tuple[float, float]] = {
    "northeast": (40.7128, -74.0060),  # NYC
    "southeast": (33.7490, -84.3880),  # Atlanta
    "midwest": (41.8781, -87.6298),    # Chicago
    "southwest": (33.4484, -112.0740), # Phoenix
    "west": (34.0522, -118.2437),      # LA
}


class RegionalPredictor:
    """Predictor using region-specific transformer models.

    Uses the appropriate regional model based on location.
    Falls back to nearest region if city not in training set.

    Example:
        >>> predictor = RegionalPredictor()
        >>> result = predictor.predict_city("Chicago", "IL")
        >>> print(f"High: {result.high_temp}F, Region: {result.region}")
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
    ):
        """Initialize the regional predictor.

        Args:
            models_dir: Directory containing regional model files
            scaler_path: Path to V3 scaler file
        """
        self.models_dir = models_dir or MODELS_DIR
        self.scaler_path = scaler_path or DATA_DIR / "scaler_v3.pkl"

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load scaler
        self._load_scaler()

        # Cache for loaded models
        self._model_cache: Dict[str, WeatherTransformer] = {}

    def _load_scaler(self) -> None:
        """Load the V3 scaler for data normalization."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"V3 scaler not found at {self.scaler_path}. "
                "Run preprocessing first to create the scaler."
            )

        scaler_info = load_scaler_v3()
        self.input_scaler = scaler_info["input_scaler"]
        self.output_scaler = scaler_info["output_scaler"]

    def _get_model_path(self, region: str) -> Path:
        """Get the model file path for a region."""
        return self.models_dir / f"region_{region}.pt"

    def _load_model(self, region: str) -> Optional[WeatherTransformer]:
        """Load a regional model from disk.

        Args:
            region: Region name (e.g., "midwest")

        Returns:
            Loaded model or None if model file doesn't exist
        """
        # Check cache first
        if region in self._model_cache:
            return self._model_cache[region]

        model_path = self._get_model_path(region)

        if not model_path.exists():
            return None

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get model config from checkpoint
        config = checkpoint.get("config", {})
        input_size = config.get("input_size", len(INPUT_FEATURE_NAMES))

        # Create model with saved config
        model = WeatherTransformer(
            input_size=input_size,
            output_size=3,
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 4),
            dropout=0.1,
            input_hours=config.get("input_hours", 72),
            output_hours=config.get("output_hours", 24),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        # Cache the model
        self._model_cache[region] = model

        return model

    def get_available_regions(self) -> List[str]:
        """Get list of regions with trained models."""
        available = []
        for region in get_all_regions():
            if self._get_model_path(region).exists():
                available.append(region)
        return available

    def _find_nearest_region(self, lat: float, lon: float) -> str:
        """Find the nearest region based on coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Name of the nearest region
        """
        min_distance = float("inf")
        nearest_region = "midwest"  # Default fallback

        for region, (center_lat, center_lon) in REGION_CENTERS.items():
            distance = _haversine_distance(lat, lon, center_lat, center_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_region = region

        return nearest_region

    def _fetch_recent_data(self, lat: float, lon: float) -> Optional[np.ndarray]:
        """Fetch the most recent 72 hours of weather data with V3 features.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            numpy array of shape (72, 11) with V3 input features, or None if failed
        """
        # Get historical data for the past 4 days to ensure we have 72 hours
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=4)

        try:
            historical = get_historical(lat, lon, start_date, end_date)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

        # Need at least 72 hours of data
        if len(historical.temperatures) < SEQUENCE_INPUT_HOURS:
            return None

        # Extract raw features (take last 72 hours)
        temps = np.array(historical.temperatures[-SEQUENCE_INPUT_HOURS:])
        humidity = np.array(historical.humidity[-SEQUENCE_INPUT_HOURS:])
        wind_speed = np.array(historical.wind_speed[-SEQUENCE_INPUT_HOURS:])
        precipitation = np.array(historical.precipitation[-SEQUENCE_INPUT_HOURS:])

        # V3 features
        wind_direction = np.array(historical.wind_direction[-SEQUENCE_INPUT_HOURS:])
        pressure_msl = np.array(historical.pressure_msl[-SEQUENCE_INPUT_HOURS:])
        dewpoint = np.array(historical.dewpoint[-SEQUENCE_INPUT_HOURS:])
        cloud_cover = np.array(historical.cloud_cover[-SEQUENCE_INPUT_HOURS:])

        # Handle missing values
        wind_direction = np.nan_to_num(wind_direction, nan=0.0)
        pressure_msl = np.nan_to_num(pressure_msl, nan=1013.25)  # Standard pressure
        dewpoint = np.nan_to_num(dewpoint, nan=temps.mean() - 10)

        # Engineer V3 features
        wind_dir_sin, wind_dir_cos = encode_wind_direction(wind_direction)
        temp_dewpoint_spread = calculate_temp_dewpoint_spread(temps, dewpoint)
        pressure_tendency = calculate_pressure_tendency(pressure_msl, hours=3)

        # Stack features in correct order matching INPUT_FEATURE_NAMES:
        # temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos, precipitation,
        # pressure_msl, dewpoint, cloud_cover, temp_dewpoint_spread, pressure_tendency
        data = np.column_stack([
            temps,
            humidity,
            wind_speed,
            wind_dir_sin,
            wind_dir_cos,
            precipitation,
            pressure_msl,
            dewpoint,
            cloud_cover,
            temp_dewpoint_spread,
            pressure_tendency,
        ])

        return data

    def _predict_with_model(
        self,
        model: WeatherTransformer,
        input_data: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Make prediction using a regional model.

        Args:
            model: The regional transformer model
            input_data: Input data of shape (72, 11) in original scale

        Returns:
            Tuple of (predictions array of shape (24, 3), confidence score)
        """
        # Normalize input
        input_normalized = self.input_scaler.transform(input_data)

        # Convert to tensor and add batch dimension
        x = torch.tensor(input_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(x)
            predictions = output.cpu().numpy()[0]  # Remove batch dimension

        # Inverse transform to original scale
        predictions_original = self.output_scaler.inverse_transform(predictions)

        # Calculate confidence based on prediction variance across time
        # Lower variance = higher confidence
        temp_std = np.std(predictions_original[:, 0])
        confidence = max(0.3, min(1.0, 1.0 - temp_std / 20.0))

        return predictions_original, confidence

    def predict(self, lat: float, lon: float) -> Optional[RegionalPrediction]:
        """Make a 24-hour prediction for coordinates using regional model.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            RegionalPrediction or None if prediction fails
        """
        # Determine region - first try to find by nearest known city
        region = self._find_nearest_region(lat, lon)

        # Check if regional model exists
        model = self._load_model(region)
        if model is None:
            # Try to find any available region
            available = self.get_available_regions()
            if not available:
                print("No regional models available")
                return None
            region = available[0]
            model = self._load_model(region)
            if model is None:
                return None

        # Fetch recent weather data
        input_data = self._fetch_recent_data(lat, lon)
        if input_data is None:
            print("Failed to fetch recent weather data")
            return None

        # Make prediction
        predictions, confidence = self._predict_with_model(model, input_data)

        # Extract individual predictions
        # Output features: temp (C), precipitation, wind_speed (km/h)
        temps_c = predictions[:, 0]
        precip = predictions[:, 1]
        winds_kmh = predictions[:, 2]

        # Convert units
        temps_f = [_celsius_to_fahrenheit(t) for t in temps_c]
        winds_mph = [_kmh_to_mph(w) for w in winds_kmh]
        precip_probs = [min(100, max(0, p * 100)) for p in precip]

        return RegionalPrediction(
            location=f"({lat:.4f}, {lon:.4f})",
            region=region,
            lat=lat,
            lon=lon,
            generated_at=datetime.now(),
            temperatures=temps_f,
            precip_probabilities=precip_probs,
            wind_speeds=winds_mph,
            high_temp=round(max(temps_f), 1),
            low_temp=round(min(temps_f), 1),
            max_wind=round(max(winds_mph), 1),
            avg_precip_prob=round(sum(precip_probs) / len(precip_probs), 1),
            model_path=str(self._get_model_path(region)),
            confidence=round(confidence, 2),
        )

    def predict_city(
        self,
        city: str,
        state: Optional[str] = None,
    ) -> Optional[RegionalPrediction]:
        """Make a 24-hour prediction for a US city.

        Automatically resolves the correct regional model based on city location.

        Args:
            city: City name (e.g., "Chicago", "New York")
            state: Optional state abbreviation (e.g., "IL", "NY")

        Returns:
            RegionalPrediction or None if prediction fails
        """
        # Normalize city name for region lookup
        city_lookup = city
        if state:
            city_lookup = f"{city}, {state}"

        # Try to get region directly from city
        region = get_region(city)
        if region is None and state:
            region = get_region(city_lookup)

        # Get coordinates
        coords = get_coordinates(city, state)
        if coords is None:
            print(f"Could not geocode: {city}, {state}")
            return None

        lat, lon = coords

        # If we found the region by city name, try to load that model
        if region is not None:
            model = self._load_model(region)
            if model is not None:
                # Use specific regional model
                input_data = self._fetch_recent_data(lat, lon)
                if input_data is None:
                    return None

                predictions, confidence = self._predict_with_model(model, input_data)

                temps_c = predictions[:, 0]
                precip = predictions[:, 1]
                winds_kmh = predictions[:, 2]

                temps_f = [_celsius_to_fahrenheit(t) for t in temps_c]
                winds_mph = [_kmh_to_mph(w) for w in winds_kmh]
                precip_probs = [min(100, max(0, p * 100)) for p in precip]

                location_name = f"{city}, {state}" if state else city

                return RegionalPrediction(
                    location=location_name,
                    region=region,
                    lat=lat,
                    lon=lon,
                    generated_at=datetime.now(),
                    temperatures=temps_f,
                    precip_probabilities=precip_probs,
                    wind_speeds=winds_mph,
                    high_temp=round(max(temps_f), 1),
                    low_temp=round(min(temps_f), 1),
                    max_wind=round(max(winds_mph), 1),
                    avg_precip_prob=round(sum(precip_probs) / len(precip_probs), 1),
                    model_path=str(self._get_model_path(region)),
                    confidence=round(confidence, 2),
                )

        # Fall back to nearest region by coordinates
        result = self.predict(lat, lon)
        if result is not None:
            # Update location name
            result.location = f"{city}, {state}" if state else city

        return result


def format_regional_prediction(pred: RegionalPrediction) -> str:
    """Format a regional prediction for display.

    Args:
        pred: RegionalPrediction to format

    Returns:
        Formatted string
    """
    lines = [
        f"Regional Forecast for {pred.location}",
        f"Region: {pred.region.upper()}",
        f"Generated: {pred.generated_at.strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        "",
        f"High Temp: {pred.high_temp}F",
        f"Low Temp: {pred.low_temp}F",
        f"Max Wind: {pred.max_wind} mph",
        f"Avg Precip Prob: {pred.avg_precip_prob}%",
        f"Confidence: {pred.confidence:.0%}",
        "",
        f"Model: {pred.model_path}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]Testing Regional Predictor[/bold cyan]\n")

    try:
        predictor = RegionalPredictor()

        # Show available regions
        available = predictor.get_available_regions()
        console.print(f"Available regional models: {available}")

        if not available:
            console.print("[yellow]No regional models trained yet.[/yellow]")
            console.print("Run: python -m src.training.regional_trainer")
        else:
            # Test with a few cities
            test_cities = [
                ("Chicago", "IL"),
                ("New York", "NY"),
                ("Los Angeles", "CA"),
                ("Miami", "FL"),
                ("Denver", "CO"),
            ]

            for city, state in test_cities:
                console.print(f"\n[bold]Predicting {city}, {state}...[/bold]")

                result = predictor.predict_city(city, state)

                if result:
                    table = Table(title=f"{result.location} ({result.region})")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")

                    table.add_row("High Temp", f"{result.high_temp}F")
                    table.add_row("Low Temp", f"{result.low_temp}F")
                    table.add_row("Max Wind", f"{result.max_wind} mph")
                    table.add_row("Avg Precip", f"{result.avg_precip_prob}%")
                    table.add_row("Confidence", f"{result.confidence:.0%}")
                    table.add_row("Region", result.region)

                    console.print(table)
                else:
                    console.print(f"[red]Failed to get prediction for {city}[/red]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Make sure to run preprocessing and training first.")

    console.print("\n[green]Test complete![/green]")
