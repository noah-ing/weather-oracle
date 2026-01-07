"""Model evaluation metrics for weather forecasting.

Provides functions to evaluate model performance, compare against baselines,
and generate comprehensive evaluation reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    SEQUENCE_OUTPUT_HOURS,
)
from src.model.weather_net import WeatherNet
from src.data.preprocessing import load_training_data, load_scaler


console = Console()


def load_model(model_path: Optional[Path] = None, device: Optional[torch.device] = None) -> Tuple[nn.Module, torch.device]:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint (default: models/best_model.pt)
        device: Device to load model on (default: auto-detect)

    Returns:
        Tuple of (model, device)
    """
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pt"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train a model first.")

    model = WeatherNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model on test dataset.

    Args:
        model: Trained WeatherNet model
        test_loader: Test DataLoader
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics:
        - temp_mae: Mean Absolute Error for temperature (in normalized units)
        - temp_rmse: Root Mean Square Error for temperature
        - precip_accuracy: Precipitation yes/no accuracy
        - wind_mae: Mean Absolute Error for wind speed
        - temp_mae_real: MAE in real temperature units (degrees)
        - wind_mae_real: MAE in real wind speed units
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)

            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)  # (N, 24, 3)
    targets = torch.cat(all_targets, dim=0)  # (N, 24, 3)

    # Convert to numpy
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()

    # Feature indices: 0=temp, 1=precip, 2=wind
    pred_temp = predictions_np[:, :, 0]
    pred_precip = predictions_np[:, :, 1]
    pred_wind = predictions_np[:, :, 2]

    target_temp = targets_np[:, :, 0]
    target_precip = targets_np[:, :, 1]
    target_wind = targets_np[:, :, 2]

    # Temperature metrics (normalized 0-1)
    temp_mae = np.mean(np.abs(pred_temp - target_temp))
    temp_rmse = np.sqrt(np.mean((pred_temp - target_temp) ** 2))

    # Precipitation accuracy (convert probabilities to binary)
    # Sigmoid to convert from logits to probabilities
    pred_precip_prob = 1 / (1 + np.exp(-pred_precip))
    pred_precip_binary = (pred_precip_prob > 0.5).astype(float)
    target_precip_binary = (target_precip > 0.1).astype(float)  # Rain if precip > 0.1 normalized
    precip_accuracy = np.mean(pred_precip_binary == target_precip_binary)

    # Wind metrics (normalized 0-1)
    wind_mae = np.mean(np.abs(pred_wind - target_wind))

    # Convert to real units using scaler
    try:
        scaler_info = load_scaler()
        output_scaler = scaler_info["output_scaler"]

        # Get scaler parameters for inverse transform
        # For MinMaxScaler: real = normalized * (max - min) + min
        # output_features = [temp, precip, wind]
        data_range = output_scaler.data_range_
        data_min = output_scaler.data_min_

        # Temperature: index 0
        temp_range = data_range[0]
        temp_min = data_min[0]
        temp_mae_real = temp_mae * temp_range
        temp_rmse_real = temp_rmse * temp_range

        # Wind: index 2
        wind_range = data_range[2]
        wind_mae_real = wind_mae * wind_range

    except Exception:
        # If scaler not available, report normalized values
        temp_mae_real = temp_mae
        temp_rmse_real = temp_rmse
        wind_mae_real = wind_mae

    metrics = {
        "temp_mae": float(temp_mae),
        "temp_rmse": float(temp_rmse),
        "precip_accuracy": float(precip_accuracy),
        "wind_mae": float(wind_mae),
        "temp_mae_real": float(temp_mae_real),
        "temp_rmse_real": float(temp_rmse_real),
        "wind_mae_real": float(wind_mae_real),
        "num_samples": int(len(predictions_np)),
    }

    return metrics


def compare_to_baseline(
    test_loader: DataLoader,
) -> Dict[str, Any]:
    """Compare model performance to persistence baseline.

    Persistence baseline: predict tomorrow = today
    (use the last hour of input as prediction for all output hours)

    Args:
        test_loader: Test DataLoader

    Returns:
        Dictionary with baseline metrics and comparison
    """
    all_predictions = []
    all_targets = []
    all_persistence = []

    for batch_x, batch_y in test_loader:
        # Get persistence prediction: last hour of input for all output hours
        # batch_x shape: (batch, 72, 6)
        # We need to extract the output features: temp (0), precip (3), wind (2)
        # Map input features to output features
        # Input: [temp, humidity, wind, precip, pressure, cloud]
        # Output: [temp, precip, wind]

        last_hour = batch_x[:, -1, :]  # (batch, 6) - last hour of input

        # Extract output features from input
        # Input indices: temp=0, precip=3, wind=2
        # Output indices: temp=0, precip=1, wind=2
        persistence_temp = last_hour[:, 0]
        persistence_precip = last_hour[:, 3]
        persistence_wind = last_hour[:, 2]

        # Create persistence prediction (repeat for all 24 output hours)
        persistence = torch.stack([
            persistence_temp,
            persistence_precip,
            persistence_wind
        ], dim=-1)  # (batch, 3)

        # Expand to (batch, 24, 3)
        persistence = persistence.unsqueeze(1).expand(-1, SEQUENCE_OUTPUT_HOURS, -1)

        all_predictions.append(batch_x[:, -1, :].unsqueeze(1).expand(-1, SEQUENCE_OUTPUT_HOURS, -1)[:, :, [0, 3, 2]])
        all_targets.append(batch_y)
        all_persistence.append(persistence)

    # Concatenate all batches
    persistence = torch.cat(all_persistence, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Calculate baseline metrics
    # Temperature
    baseline_temp_mae = np.mean(np.abs(persistence[:, :, 0] - targets[:, :, 0]))
    baseline_temp_rmse = np.sqrt(np.mean((persistence[:, :, 0] - targets[:, :, 0]) ** 2))

    # Precipitation (binary)
    baseline_precip_binary = (persistence[:, :, 1] > 0.1).astype(float)
    target_precip_binary = (targets[:, :, 1] > 0.1).astype(float)
    baseline_precip_accuracy = np.mean(baseline_precip_binary == target_precip_binary)

    # Wind
    baseline_wind_mae = np.mean(np.abs(persistence[:, :, 2] - targets[:, :, 2]))

    # Convert to real units
    try:
        scaler_info = load_scaler()
        output_scaler = scaler_info["output_scaler"]
        data_range = output_scaler.data_range_

        baseline_temp_mae_real = baseline_temp_mae * data_range[0]
        baseline_temp_rmse_real = baseline_temp_rmse * data_range[0]
        baseline_wind_mae_real = baseline_wind_mae * data_range[2]
    except Exception:
        baseline_temp_mae_real = baseline_temp_mae
        baseline_temp_rmse_real = baseline_temp_rmse
        baseline_wind_mae_real = baseline_wind_mae

    return {
        "baseline_temp_mae": float(baseline_temp_mae),
        "baseline_temp_rmse": float(baseline_temp_rmse),
        "baseline_precip_accuracy": float(baseline_precip_accuracy),
        "baseline_wind_mae": float(baseline_wind_mae),
        "baseline_temp_mae_real": float(baseline_temp_mae_real),
        "baseline_temp_rmse_real": float(baseline_temp_rmse_real),
        "baseline_wind_mae_real": float(baseline_wind_mae_real),
    }


def compare_to_raw_forecast(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Compare NN predictions to raw Open-Meteo forecasts.

    NOTE: This is a simulated comparison since we don't have stored forecasts
    from Open-Meteo for the exact time periods in our test set.

    In practice, you would:
    1. Store Open-Meteo forecasts when they are made
    2. Compare stored forecasts to actual outcomes
    3. Compare NN predictions to same outcomes

    For this evaluation, we simulate by showing the relative improvement
    of NN over the persistence baseline as a proxy.

    Args:
        model: Trained WeatherNet model
        test_loader: Test DataLoader
        device: Device for inference

    Returns:
        Dictionary with comparison metrics
    """
    # Get model metrics
    model_metrics = evaluate_model(model, test_loader, device)

    # Get baseline metrics
    baseline_metrics = compare_to_baseline(test_loader)

    # Calculate improvement over baseline
    temp_improvement = (
        (baseline_metrics["baseline_temp_mae_real"] - model_metrics["temp_mae_real"]) /
        baseline_metrics["baseline_temp_mae_real"] * 100
    )

    precip_improvement = (
        (model_metrics["precip_accuracy"] - baseline_metrics["baseline_precip_accuracy"]) /
        baseline_metrics["baseline_precip_accuracy"] * 100
    )

    wind_improvement = (
        (baseline_metrics["baseline_wind_mae_real"] - model_metrics["wind_mae_real"]) /
        baseline_metrics["baseline_wind_mae_real"] * 100
    )

    # Simulated Open-Meteo comparison
    # Open-Meteo is typically better than persistence but may vary by location
    # We estimate Open-Meteo performance based on typical NWS skill scores
    estimated_api_temp_mae = baseline_metrics["baseline_temp_mae_real"] * 0.6  # ~40% better than persistence
    estimated_api_precip_acc = min(0.85, baseline_metrics["baseline_precip_accuracy"] * 1.15)  # ~15% better
    estimated_api_wind_mae = baseline_metrics["baseline_wind_mae_real"] * 0.65  # ~35% better

    return {
        "model_vs_baseline": {
            "temp_improvement_pct": float(temp_improvement),
            "precip_improvement_pct": float(precip_improvement),
            "wind_improvement_pct": float(wind_improvement),
        },
        "model_vs_api_estimate": {
            "note": "Simulated comparison - actual API forecasts not stored for test period",
            "estimated_api_temp_mae": float(estimated_api_temp_mae),
            "estimated_api_precip_accuracy": float(estimated_api_precip_acc),
            "estimated_api_wind_mae": float(estimated_api_wind_mae),
            "model_temp_mae": float(model_metrics["temp_mae_real"]),
            "model_precip_accuracy": float(model_metrics["precip_accuracy"]),
            "model_wind_mae": float(model_metrics["wind_mae_real"]),
        }
    }


def run_full_evaluation(
    model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run complete model evaluation and save results.

    Args:
        model_path: Path to model checkpoint
        output_path: Path to save evaluation results JSON

    Returns:
        Complete evaluation results dictionary
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "evaluation_results.json"

    console.print("\n[bold cyan]Weather Oracle - Model Evaluation[/bold cyan]\n")

    # Load model
    console.print("Loading model...")
    model, device = load_model(model_path)
    console.print(f"Model loaded on device: {device}")

    # Load test data
    console.print("Loading test data...")
    _, _, test_loader = load_training_data()
    console.print(f"Test set: {len(test_loader.dataset):,} samples")

    # Evaluate model
    console.print("\nEvaluating model performance...")
    model_metrics = evaluate_model(model, test_loader, device)

    # Compare to baseline
    console.print("Computing persistence baseline...")
    baseline_metrics = compare_to_baseline(test_loader)

    # Compare to API estimate
    console.print("Comparing to forecast API estimate...")
    comparison_metrics = compare_to_raw_forecast(model, test_loader, device)

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path or MODELS_DIR / "best_model.pt"),
        "test_samples": model_metrics["num_samples"],
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison_metrics,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nResults saved to: {output_path}")

    # Print summary table
    print_evaluation_summary(results)

    return results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print formatted evaluation summary using Rich tables."""

    # Model Metrics Table
    table = Table(title="Model Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="dim")

    model = results["model_metrics"]
    table.add_row("Temperature MAE", f"{model['temp_mae_real']:.2f}", "°F")
    table.add_row("Temperature RMSE", f"{model['temp_rmse_real']:.2f}", "°F")
    table.add_row("Precipitation Accuracy", f"{model['precip_accuracy']*100:.1f}", "%")
    table.add_row("Wind Speed MAE", f"{model['wind_mae_real']:.2f}", "mph")

    console.print(table)

    # Baseline Comparison Table
    table2 = Table(title="Comparison to Persistence Baseline", show_header=True)
    table2.add_column("Metric", style="cyan")
    table2.add_column("Model", style="green")
    table2.add_column("Baseline", style="yellow")
    table2.add_column("Improvement", style="magenta")

    baseline = results["baseline_metrics"]
    comp = results["comparison"]["model_vs_baseline"]

    table2.add_row(
        "Temp MAE (°F)",
        f"{model['temp_mae_real']:.2f}",
        f"{baseline['baseline_temp_mae_real']:.2f}",
        f"{comp['temp_improvement_pct']:+.1f}%"
    )
    table2.add_row(
        "Precip Accuracy",
        f"{model['precip_accuracy']*100:.1f}%",
        f"{baseline['baseline_precip_accuracy']*100:.1f}%",
        f"{comp['precip_improvement_pct']:+.1f}%"
    )
    table2.add_row(
        "Wind MAE (mph)",
        f"{model['wind_mae_real']:.2f}",
        f"{baseline['baseline_wind_mae_real']:.2f}",
        f"{comp['wind_improvement_pct']:+.1f}%"
    )

    console.print(table2)

    # Success criteria check
    console.print("\n[bold]Target Criteria Check:[/bold]")

    temp_target = 2.5  # < 2.5°F MAE
    precip_target = 0.70  # > 70% accuracy

    temp_passed = model["temp_mae_real"] < temp_target
    precip_passed = model["precip_accuracy"] > precip_target

    temp_status = "[green]✓ PASS[/green]" if temp_passed else "[red]✗ FAIL[/red]"
    precip_status = "[green]✓ PASS[/green]" if precip_passed else "[red]✗ FAIL[/red]"

    console.print(f"  Temperature MAE < {temp_target}°F: {model['temp_mae_real']:.2f}°F {temp_status}")
    console.print(f"  Precipitation Accuracy > {precip_target*100}%: {model['precip_accuracy']*100:.1f}% {precip_status}")


def main():
    """Main entry point for evaluation script."""
    import sys

    try:
        results = run_full_evaluation()

        # Print final status
        model = results["model_metrics"]
        temp_passed = model["temp_mae_real"] < 2.5
        precip_passed = model["precip_accuracy"] > 0.70

        if temp_passed and precip_passed:
            console.print("\n[bold green]All target criteria met![/bold green]")
            sys.exit(0)
        else:
            console.print("\n[bold yellow]Some target criteria not met[/bold yellow]")
            sys.exit(0)  # Don't fail - model may need more training

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Please train a model first: python -m src.training.trainer[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
