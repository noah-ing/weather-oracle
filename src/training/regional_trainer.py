"""Regional model training pipeline for Weather Oracle V3.

Trains separate WeatherTransformer models for each US climate region,
enabling specialized predictions based on local weather patterns.
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.config import (
    MODELS_DIR,
    PROJECT_ROOT,
    LEARNING_RATE,
    BATCH_SIZE,
)
from src.model.weather_transformer import WeatherTransformer, count_parameters
from src.data.preprocessing import load_training_data, INPUT_FEATURE_NAMES
from src.data.regions import get_all_regions, get_formatted_cities_in_region


console = Console()


class WeatherLoss(nn.Module):
    """Combined loss for weather prediction.

    Uses MSE for temperature and wind speed, BCE for precipitation probability.
    """

    def __init__(
        self,
        temp_weight: float = 1.0,
        precip_weight: float = 0.5,
        wind_weight: float = 0.5,
    ):
        """Initialize loss function.

        Args:
            temp_weight: Weight for temperature MSE loss
            precip_weight: Weight for precipitation BCE loss
            wind_weight: Weight for wind speed MSE loss
        """
        super().__init__()
        self.temp_weight = temp_weight
        self.precip_weight = precip_weight
        self.wind_weight = wind_weight

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            predictions: Model predictions, shape (batch, seq_len, 3)
            targets: Ground truth, shape (batch, seq_len, 3)

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Extract individual predictions (indices: 0=temp, 1=precip, 2=wind)
        pred_temp = predictions[:, :, 0]
        pred_precip = predictions[:, :, 1]
        pred_wind = predictions[:, :, 2]

        target_temp = targets[:, :, 0]
        target_precip = targets[:, :, 1]
        target_wind = targets[:, :, 2]

        # Compute individual losses
        temp_loss = self.mse(pred_temp, target_temp)
        target_precip_clamped = torch.clamp(target_precip, 0, 1)
        precip_loss = self.bce(pred_precip, target_precip_clamped)
        wind_loss = self.mse(pred_wind, target_wind)

        # Combine losses
        total_loss = (
            self.temp_weight * temp_loss +
            self.precip_weight * precip_loss +
            self.wind_weight * wind_loss
        )

        loss_dict = {
            "temp_loss": temp_loss.item(),
            "precip_loss": precip_loss.item(),
            "wind_loss": wind_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


def train_epoch_with_accumulation(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 4,
) -> Dict[str, float]:
    """Train for one epoch with gradient accumulation.

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        accumulation_steps: Number of steps to accumulate gradients

    Returns:
        Dictionary with average losses
    """
    model.train()
    total_losses = {"temp_loss": 0, "precip_loss": 0, "wind_loss": 0, "total_loss": 0}
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        predictions = model(batch_x)

        # Compute loss and scale for accumulation
        loss, loss_dict = criterion(predictions, batch_y)
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Accumulate losses for logging (unscaled)
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate for one epoch.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary with average losses
    """
    model.eval()
    total_losses = {"temp_loss": 0, "precip_loss": 0, "wind_loss": 0, "total_loss": 0}
    num_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss, loss_dict = criterion(predictions, batch_y)

            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1

    # Average losses
    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches

    return total_losses


def train_single_region(
    region: str,
    epochs: int = 100,
    patience: int = 15,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    accumulation_steps: int = 4,
    model_path: Optional[Path] = None,
    log_file: Optional[Any] = None,
    resume: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a WeatherTransformer model for a single region.

    Args:
        region: Region name (e.g., "midwest")
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        lr: Initial learning rate
        batch_size: Batch size for training
        accumulation_steps: Steps for gradient accumulation (effective batch = batch_size * accumulation_steps)
        model_path: Path to save best model
        log_file: CSV file handle for logging (optional)

    Returns:
        model: Trained model
        history: Training history dictionary
    """
    # Set up paths
    if model_path is None:
        model_path = MODELS_DIR / f"region_{region}.pt"

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get region cities for logging
    region_cities = get_formatted_cities_in_region(region)
    console.print(f"\n[bold cyan]Training model for region: {region}[/bold cyan]")
    console.print(f"Cities: {', '.join(region_cities)}")
    console.print(f"Device: {device}")

    # Load data for this region
    try:
        train_loader, val_loader, test_loader = load_training_data(
            batch_size=batch_size,
            use_v3_features=True,
            region_filter=region,
        )
    except ValueError as e:
        console.print(f"[red]Error loading data for {region}: {e}[/red]")
        return None, {}

    if len(train_loader) == 0:
        console.print(f"[red]No training data for {region}[/red]")
        return None, {}

    # Create model - V3 uses 11 input features
    num_input_features = len(INPUT_FEATURE_NAMES)
    model = WeatherTransformer(
        input_size=num_input_features,
        output_size=3,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        input_hours=72,  # V3 uses 72h input (matching preprocessing)
        output_hours=24,
    ).to(device)

    console.print(f"Model parameters: {count_parameters(model):,}")

    # Loss function
    criterion = WeatherLoss()

    # Optimizer - AdamW with weight decay
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    # History
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    # Resume from checkpoint if requested
    if resume and model_path.exists():
        console.print(f"[yellow]Resuming from checkpoint: {model_path}[/yellow]")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        history = checkpoint.get("history", history)
        console.print(f"[yellow]Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}[/yellow]")

    effective_batch_size = batch_size * accumulation_steps
    console.print(f"Training for up to {epochs} epochs (effective batch size: {effective_batch_size})")
    console.print(f"Early stopping patience: {patience}")

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Train with gradient accumulation
        train_losses = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, device, accumulation_steps
        )

        # Validate
        val_losses = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_losses["total_loss"])
        history["val_loss"].append(val_losses["total_loss"])
        history["lr"].append(current_lr)

        # Log to CSV if file handle provided
        if log_file is not None:
            log_file.writerow([
                datetime.now().isoformat(),
                region,
                epoch,
                train_losses["total_loss"],
                val_losses["total_loss"],
                train_losses["temp_loss"],
                val_losses["temp_loss"],
                train_losses["precip_loss"],
                val_losses["precip_loss"],
                train_losses["wind_loss"],
                val_losses["wind_loss"],
                current_lr,
                best_val_loss,
            ])

        # Check for improvement
        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "history": history,
                "region": region,
                "config": {
                    "input_size": num_input_features,
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 4,
                    "input_hours": 72,
                    "output_hours": 24,
                },
            }, model_path)

            improvement_str = "[green]* Best[/green]"
        else:
            epochs_without_improvement += 1
            improvement_str = ""

        # Print epoch progress
        epoch_time = time.time() - epoch_start
        console.print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_losses['total_loss']:.4f} | "
            f"Val: {val_losses['total_loss']:.4f} | "
            f"Best: {best_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s {improvement_str}"
        )

        # Early stopping check
        if epochs_without_improvement >= patience:
            console.print(f"  [yellow]Early stopping triggered after {epoch} epochs[/yellow]")
            break

    total_time = time.time() - start_time
    console.print(f"  [green]Training complete![/green] Time: {total_time / 60:.1f} min, Best val loss: {best_val_loss:.4f}")

    # Load best model for return
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    history["total_time"] = total_time
    history["best_val_loss"] = best_val_loss
    history["final_epoch"] = epoch

    return model, history


def train_regional_models(
    epochs: int = 100,
    patience: int = 15,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    accumulation_steps: int = 4,
    regions: Optional[list] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Train WeatherTransformer models for all regions.

    Args:
        epochs: Maximum number of training epochs per region
        patience: Early stopping patience
        lr: Initial learning rate
        batch_size: Batch size for training
        accumulation_steps: Steps for gradient accumulation
        regions: List of regions to train (default: all regions)
        log_path: Path to save training log CSV

    Returns:
        Dictionary mapping region name to training history
    """
    if regions is None:
        regions = get_all_regions()

    if log_path is None:
        log_path = PROJECT_ROOT / "training_log_v3.csv"

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Weather Oracle V3 - Regional Model Training[/bold cyan]")
    console.print("=" * 60)
    console.print(f"\nRegions to train: {', '.join(regions)}")
    console.print(f"Epochs per region: {epochs}")
    console.print(f"Early stopping patience: {patience}")
    console.print(f"Learning rate: {lr}")
    console.print(f"Batch size: {batch_size} (effective: {batch_size * accumulation_steps})")
    console.print(f"Log file: {log_path}")

    # Initialize CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "region", "epoch", "train_loss", "val_loss",
            "train_temp_loss", "val_temp_loss", "train_precip_loss", "val_precip_loss",
            "train_wind_loss", "val_wind_loss", "lr", "best_val_loss",
        ])

    # Train each region
    results = {}
    total_start_time = time.time()

    for idx, region in enumerate(regions, 1):
        console.print(f"\n[bold]({idx}/{len(regions)}) Training {region.upper()} region[/bold]")

        # Open log file in append mode for this region
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)

            model, history = train_single_region(
                region=region,
                epochs=epochs,
                patience=patience,
                lr=lr,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                log_file=writer,
            )

            if model is not None:
                results[region] = history
            else:
                results[region] = {"error": "Training failed"}

    total_time = time.time() - total_start_time

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Training Complete - Summary[/bold green]")
    console.print("=" * 60)

    table = Table(title="Regional Model Results")
    table.add_column("Region", style="cyan")
    table.add_column("Best Val Loss", justify="right")
    table.add_column("Epochs", justify="right")
    table.add_column("Time (min)", justify="right")
    table.add_column("Status", justify="center")

    for region, history in results.items():
        if "error" in history:
            table.add_row(region, "-", "-", "-", "[red]Failed[/red]")
        else:
            table.add_row(
                region,
                f"{history['best_val_loss']:.4f}",
                str(history['final_epoch']),
                f"{history['total_time'] / 60:.1f}",
                "[green]Done[/green]",
            )

    console.print(table)
    console.print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    console.print(f"Models saved to: {MODELS_DIR}")
    console.print(f"Training log: {log_path}")

    return results


def main():
    """Main entry point for regional training."""
    # Default parameters
    epochs = 100
    patience = 15

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            console.print("[red]Usage: python -m src.training.regional_trainer [epochs] [patience][/red]")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            patience = int(sys.argv[2])
        except ValueError:
            console.print("[red]Usage: python -m src.training.regional_trainer [epochs] [patience][/red]")
            sys.exit(1)

    train_regional_models(epochs=epochs, patience=patience)


if __name__ == "__main__":
    main()
