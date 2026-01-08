"""Multi-model ensemble predictor (V2/V3).

Combines forecasts from multiple sources with intelligent weighting:
- Regional NN (V3): Transformer models trained on location-specific patterns
- Neural network model (V1): LSTM model from trained WeatherNet
- GFS, ECMWF, ICON, GEM (from Open-Meteo)
- NWS (National Weather Service)

Features:
- Inverse MAE weighting: more accurate sources get higher weight
- Bias correction: adjusts each source for systematic errors
- Uncertainty estimation: std from model disagreement
- Per-source contribution tracking
- Regional NN gets higher base weight (0.25) due to location-specific training

Usage:
    from src.inference.ensemble_v2 import predict_ensemble

    result = predict_ensemble("NYC")
    print(f"High: {result.high_temp}F +/- {result.high_std}F")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {[c.source for c in result.contributions]}")
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
import statistics

from src.api.geocoding import get_coordinates
from src.api.weather_models import (
    get_all_forecasts,
    get_open_meteo_forecast,
    get_nws_forecast,
    WeatherForecast,
)
from src.calibration.bias_correction import correct_forecast, get_bias_correction
from src.tracking.forecast_tracker import get_accuracy_by_source, AccuracyReport


# Default weights when no accuracy data is available
DEFAULT_WEIGHTS = {
    "regional_nn": 0.25,  # Regional transformer model (V3) - location-specific training
    "nws": 0.20,          # NWS is authoritative for US
    "gfs": 0.15,          # GFS is NOAA's global model
    "ecmwf": 0.15,        # ECMWF is very accurate globally
    "icon": 0.10,         # DWD ICON
    "gem": 0.05,          # Canadian GEM
    "nn_model": 0.10,     # Our trained LSTM neural network
}

# Minimum weight to prevent any source from being completely ignored
MIN_WEIGHT = 0.05

# Maximum weight to prevent over-reliance on single source
MAX_WEIGHT = 0.50


@dataclass
class SourceContribution:
    """Contribution from a single forecast source."""
    source: str
    raw_high: float          # Raw forecast high temp (F)
    raw_low: float           # Raw forecast low temp (F)
    corrected_high: float    # Bias-corrected high temp (F)
    corrected_low: float     # Bias-corrected low temp (F)
    weight: float            # Weight used in ensemble (0-1)
    mae: Optional[float]     # Historical MAE if available
    bias_high: Optional[float]  # Bias correction applied to high
    bias_low: Optional[float]   # Bias correction applied to low


@dataclass
class EnsemblePrediction:
    """Ensemble weather prediction with uncertainty estimates.

    Combines multiple forecast sources into a single prediction with
    confidence intervals based on model disagreement.
    """
    location: str
    lat: float
    lon: float
    target_date: date
    prediction_time: datetime

    # Ensemble predictions
    high_temp: float          # Ensemble mean high temp (F)
    low_temp: float           # Ensemble mean low temp (F)
    high_std: float           # Std dev of high temp predictions
    low_std: float            # Std dev of low temp predictions

    # Confidence (0-1) based on model agreement
    confidence: float

    # Weather details
    precip_probability: float  # 0-100%
    wind_speed_max: float      # mph
    conditions: str

    # Per-source contributions
    contributions: List[SourceContribution] = field(default_factory=list)

    # Summary of sources used
    sources_used: List[str] = field(default_factory=list)
    total_sources: int = 0


def _normalize_location(location: str) -> str:
    """Normalize location name for database lookups.

    Args:
        location: Location string (e.g., "NYC", "New York, NY")

    Returns:
        Normalized location name
    """
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


def _get_source_weights(
    location: str,
    sources: List[str],
    days: int = 7,
) -> Dict[str, float]:
    """Calculate inverse MAE weights for each source.

    Sources with lower MAE (better accuracy) get higher weights.

    Args:
        location: Location name for accuracy lookup
        sources: List of source names to weight
        days: Rolling window for accuracy calculation

    Returns:
        Dict mapping source -> weight (0-1), normalized to sum to 1.0
    """
    # Try to get accuracy data
    location_normalized = _normalize_location(location)
    accuracy_data = get_accuracy_by_source(location=location_normalized, days=days)

    # If no accuracy data, use defaults
    if not accuracy_data or location_normalized not in accuracy_data:
        return {s: DEFAULT_WEIGHTS.get(s, 0.1) for s in sources}

    location_accuracy = accuracy_data[location_normalized]

    # Calculate inverse MAE weights
    mae_values: Dict[str, float] = {}

    for source in sources:
        if source in location_accuracy:
            report = location_accuracy[source]
            # Use combined MAE (average of high and low MAE)
            mae = report.mae_combined if report.mae_combined > 0 else 1.0
            mae_values[source] = mae
        else:
            # Use default MAE estimate for sources without data
            mae_values[source] = 3.0  # Assume 3F MAE as default

    # Convert MAE to weights (inverse relationship)
    # Lower MAE = higher weight
    inverse_mae = {s: 1.0 / mae for s, mae in mae_values.items()}
    total_inverse = sum(inverse_mae.values())

    if total_inverse < 0.01:
        return {s: DEFAULT_WEIGHTS.get(s, 0.1) for s in sources}

    # Normalize weights to sum to 1.0
    weights = {s: inv / total_inverse for s, inv in inverse_mae.items()}

    # Clamp weights to min/max range
    for source in weights:
        weights[source] = max(MIN_WEIGHT, min(MAX_WEIGHT, weights[source]))

    # Re-normalize after clamping
    total = sum(weights.values())
    weights = {s: w / total for s, w in weights.items()}

    return weights


def _apply_bias_correction(
    source: str,
    location: str,
    high_temp: float,
    low_temp: float,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """Apply bias correction to a forecast.

    Args:
        source: Forecast source name
        location: Location name
        high_temp: Raw high temp prediction
        low_temp: Raw low temp prediction

    Returns:
        Tuple of (corrected_high, corrected_low, bias_high, bias_low)
    """
    # Get bias correction if available
    bias = get_bias_correction(source, _normalize_location(location))

    if bias is None:
        # Try without normalization
        bias = get_bias_correction(source, location)

    if bias is None:
        return high_temp, low_temp, None, None

    # Apply correction: subtract bias
    corrected_high = high_temp - bias.bias_high
    corrected_low = low_temp - bias.bias_low

    return corrected_high, corrected_low, bias.bias_high, bias.bias_low


def _get_nn_forecast(lat: float, lon: float) -> Optional[WeatherForecast]:
    """Get forecast from our trained neural network model.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        WeatherForecast from NN model, or None if unavailable
    """
    try:
        from src.inference.predictor import WeatherPredictor

        predictor = WeatherPredictor()
        forecast = predictor.predict(lat, lon)

        if forecast is None or not forecast.hourly:
            return None

        # Extract high/low from hourly predictions
        temps = [h.temperature for h in forecast.hourly]
        precip_probs = [h.precip_probability for h in forecast.hourly]
        wind_speeds = [h.wind_speed for h in forecast.hourly]

        target_date = (datetime.now() + timedelta(days=1)).date()

        return WeatherForecast(
            source="nn_model",
            location=f"{lat:.2f},{lon:.2f}",
            lat=lat,
            lon=lon,
            forecast_time=datetime.now(),
            target_date=target_date,
            high_temp=max(temps),
            low_temp=min(temps),
            precip_probability=max(precip_probs) if precip_probs else 0,
            precip_amount=0.0,
            wind_speed_max=max(wind_speeds) if wind_speeds else 0,
            conditions=forecast.hourly[12].conditions if len(forecast.hourly) > 12 else "",
            hourly_temps=temps,
            hourly_precip_prob=precip_probs,
            hourly_wind=wind_speeds,
            confidence=None,  # NN model doesn't provide confidence directly
        )
    except Exception as e:
        print(f"NN model forecast error: {e}")
        return None


def _get_regional_nn_forecast(
    lat: float,
    lon: float,
    city: Optional[str] = None,
    state: Optional[str] = None,
) -> Optional[WeatherForecast]:
    """Get forecast from regional transformer model (V3).

    Uses region-specific transformer models trained on local weather patterns.
    Falls back to nearest region if exact city not in training set.

    Args:
        lat: Latitude
        lon: Longitude
        city: Optional city name for region lookup
        state: Optional state abbreviation

    Returns:
        WeatherForecast from regional model, or None if unavailable
    """
    try:
        from src.inference.regional_predictor import RegionalPredictor

        predictor = RegionalPredictor()

        # Check if any regional models are available
        available_regions = predictor.get_available_regions()
        if not available_regions:
            return None

        # Try city-based prediction first, fall back to coordinates
        if city:
            forecast = predictor.predict_city(city, state)
        else:
            forecast = predictor.predict(lat, lon)

        if forecast is None:
            return None

        target_date = (datetime.now() + timedelta(days=1)).date()

        return WeatherForecast(
            source="regional_nn",
            location=forecast.location,
            lat=lat,
            lon=lon,
            forecast_time=datetime.now(),
            target_date=target_date,
            high_temp=forecast.high_temp,
            low_temp=forecast.low_temp,
            precip_probability=forecast.avg_precip_prob,
            precip_amount=0.0,
            wind_speed_max=forecast.max_wind,
            conditions=f"Region: {forecast.region}",
            hourly_temps=forecast.temperatures,
            hourly_precip_prob=forecast.precip_probabilities,
            hourly_wind=forecast.wind_speeds,
            confidence=forecast.confidence,
        )
    except FileNotFoundError:
        # V3 scaler or models not available
        return None
    except Exception as e:
        print(f"Regional NN forecast error: {e}")
        return None


def predict_ensemble(
    location: str,
    include_nn: bool = True,
    include_backup: bool = False,
) -> Optional[EnsemblePrediction]:
    """Generate an ensemble prediction for a location.

    Combines forecasts from multiple sources with intelligent weighting
    based on historical accuracy and bias correction.

    Args:
        location: City name or abbreviation (e.g., "NYC", "New York", "Chicago, IL")
        include_nn: Whether to include neural network model
        include_backup: Whether to include backup API sources

    Returns:
        EnsemblePrediction with mean, std, and per-source contributions

    Example:
        >>> result = predict_ensemble("NYC")
        >>> print(f"High: {result.high_temp}F +/- {result.high_std}F")
    """
    # Normalize and geocode location
    location_normalized = _normalize_location(location)

    # Parse city and state
    parts = location_normalized.split(",")
    city = parts[0].strip()
    state = parts[1].strip() if len(parts) > 1 else None

    coords = get_coordinates(city, state)
    if coords is None:
        print(f"Could not geocode: {location}")
        return None

    lat, lon = coords

    # Collect forecasts from all sources
    forecasts: Dict[str, WeatherForecast] = {}

    # Get external API forecasts (GFS, ECMWF, ICON, GEM, NWS)
    api_forecasts = get_all_forecasts(lat, lon, include_backup=include_backup)
    forecasts.update(api_forecasts)

    # Get neural network forecast (LSTM model)
    if include_nn:
        nn_forecast = _get_nn_forecast(lat, lon)
        if nn_forecast:
            forecasts["nn_model"] = nn_forecast

    # Get regional transformer forecast (V3)
    if include_nn:
        regional_forecast = _get_regional_nn_forecast(lat, lon, city, state)
        if regional_forecast:
            forecasts["regional_nn"] = regional_forecast

    if not forecasts:
        print(f"No forecasts available for {location}")
        return None

    # Get weights based on historical accuracy
    sources = list(forecasts.keys())
    weights = _get_source_weights(location_normalized, sources)

    # Apply bias correction and collect contributions
    contributions: List[SourceContribution] = []
    corrected_highs: List[float] = []
    corrected_lows: List[float] = []
    weighted_high_sum = 0.0
    weighted_low_sum = 0.0
    weighted_precip_sum = 0.0
    weighted_wind_sum = 0.0
    total_weight = 0.0

    # Get accuracy data for MAE display
    accuracy_data = get_accuracy_by_source(location=location_normalized, days=7)
    location_accuracy = accuracy_data.get(location_normalized, {}) if accuracy_data else {}

    for source, fc in forecasts.items():
        weight = weights.get(source, DEFAULT_WEIGHTS.get(source, 0.1))

        # Apply bias correction
        corrected_high, corrected_low, bias_high, bias_low = _apply_bias_correction(
            source, location_normalized, fc.high_temp, fc.low_temp
        )

        # Get MAE if available
        mae = None
        if source in location_accuracy:
            mae = location_accuracy[source].mae_combined

        # Create contribution record (ensure all values are Python floats)
        contribution = SourceContribution(
            source=source,
            raw_high=float(fc.high_temp),
            raw_low=float(fc.low_temp),
            corrected_high=float(corrected_high),
            corrected_low=float(corrected_low),
            weight=float(weight),
            mae=float(mae) if mae is not None else None,
            bias_high=float(bias_high) if bias_high is not None else None,
            bias_low=float(bias_low) if bias_low is not None else None,
        )
        contributions.append(contribution)

        # Accumulate weighted values (convert to Python float for statistics module)
        corrected_highs.append(float(corrected_high))
        corrected_lows.append(float(corrected_low))
        weighted_high_sum += weight * corrected_high
        weighted_low_sum += weight * corrected_low
        weighted_precip_sum += weight * fc.precip_probability
        weighted_wind_sum += weight * fc.wind_speed_max
        total_weight += weight

    # Calculate ensemble mean
    if total_weight < 0.01:
        return None

    ensemble_high = weighted_high_sum / total_weight
    ensemble_low = weighted_low_sum / total_weight
    ensemble_precip = weighted_precip_sum / total_weight
    ensemble_wind = weighted_wind_sum / total_weight

    # Calculate std dev (measure of model disagreement)
    high_std = statistics.stdev(corrected_highs) if len(corrected_highs) > 1 else 0.0
    low_std = statistics.stdev(corrected_lows) if len(corrected_lows) > 1 else 0.0

    # Calculate confidence based on model agreement
    # Lower std = higher confidence
    # Std of 0-2F = high confidence (0.8-1.0)
    # Std of 2-5F = medium confidence (0.5-0.8)
    # Std of 5+F = low confidence (<0.5)
    avg_std = (high_std + low_std) / 2
    if avg_std <= 2.0:
        confidence = 0.8 + 0.2 * (1.0 - avg_std / 2.0)
    elif avg_std <= 5.0:
        confidence = 0.5 + 0.3 * (1.0 - (avg_std - 2.0) / 3.0)
    else:
        confidence = max(0.2, 0.5 - 0.1 * (avg_std - 5.0))

    confidence = max(0.1, min(1.0, confidence))

    # Get conditions from majority vote or first available
    conditions_list = [fc.conditions for fc in forecasts.values() if fc.conditions]
    conditions = conditions_list[0] if conditions_list else ""

    # Get target date from forecasts
    target_date = list(forecasts.values())[0].target_date

    return EnsemblePrediction(
        location=location_normalized,
        lat=lat,
        lon=lon,
        target_date=target_date,
        prediction_time=datetime.now(),
        high_temp=round(float(ensemble_high), 1),
        low_temp=round(float(ensemble_low), 1),
        high_std=round(float(high_std), 1),
        low_std=round(float(low_std), 1),
        confidence=round(float(confidence), 2),
        precip_probability=round(float(ensemble_precip), 0),
        wind_speed_max=round(float(ensemble_wind), 1),
        conditions=conditions,
        contributions=sorted(contributions, key=lambda c: c.weight, reverse=True),
        sources_used=sources,
        total_sources=len(sources),
    )


def format_ensemble_prediction(prediction: EnsemblePrediction) -> str:
    """Format an ensemble prediction for display.

    Args:
        prediction: EnsemblePrediction to format

    Returns:
        Formatted string
    """
    lines = [
        f"Ensemble Forecast for {prediction.location}",
        f"Target Date: {prediction.target_date}",
        "=" * 50,
        "",
        f"High Temp: {prediction.high_temp}F +/- {prediction.high_std}F",
        f"Low Temp:  {prediction.low_temp}F +/- {prediction.low_std}F",
        f"Confidence: {prediction.confidence:.0%}",
        "",
        f"Precip Probability: {prediction.precip_probability:.0f}%",
        f"Max Wind: {prediction.wind_speed_max:.0f} mph",
        f"Conditions: {prediction.conditions}",
        "",
        "Source Contributions:",
        "-" * 50,
        f"{'Source':<12} {'Weight':>8} {'Raw High':>10} {'Corrected':>10} {'MAE':>8}",
    ]

    for c in prediction.contributions:
        mae_str = f"{c.mae:.1f}F" if c.mae else "N/A"
        lines.append(
            f"{c.source:<12} {c.weight:>7.1%} {c.raw_high:>9.1f}F {c.corrected_high:>9.1f}F {mae_str:>8}"
        )

    lines.append("")
    lines.append(f"Sources used: {prediction.total_sources} ({', '.join(prediction.sources_used)})")

    return "\n".join(lines)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]Testing Ensemble V2 Predictor[/bold cyan]\n")

    # Test with a few cities
    test_cities = ["NYC", "Chicago, IL", "Denver"]

    for city in test_cities:
        console.print(f"\n[bold]Forecasting {city}...[/bold]")

        prediction = predict_ensemble(city)

        if prediction:
            table = Table(title=f"Ensemble Forecast: {prediction.location}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Target Date", str(prediction.target_date))
            table.add_row("High Temp", f"{prediction.high_temp}F +/- {prediction.high_std}F")
            table.add_row("Low Temp", f"{prediction.low_temp}F +/- {prediction.low_std}F")
            table.add_row("Confidence", f"{prediction.confidence:.0%}")
            table.add_row("Precip Prob", f"{prediction.precip_probability:.0f}%")
            table.add_row("Max Wind", f"{prediction.wind_speed_max:.0f} mph")
            table.add_row("Sources", f"{prediction.total_sources}")

            console.print(table)

            # Show source contributions
            contrib_table = Table(title="Source Contributions")
            contrib_table.add_column("Source")
            contrib_table.add_column("Weight")
            contrib_table.add_column("Raw High")
            contrib_table.add_column("Corrected")
            contrib_table.add_column("MAE")

            for c in prediction.contributions:
                mae_str = f"{c.mae:.1f}F" if c.mae else "N/A"
                contrib_table.add_row(
                    c.source,
                    f"{c.weight:.1%}",
                    f"{c.raw_high:.1f}F",
                    f"{c.corrected_high:.1f}F",
                    mae_str,
                )

            console.print(contrib_table)
        else:
            console.print(f"[red]Failed to get prediction for {city}[/red]")

    console.print("\n[green]Test complete![/green]")
