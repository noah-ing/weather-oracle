"""Training pipeline for WeatherNet with checkpointing and early stopping."""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.config import (
    MODELS_DIR,
    PROJECT_ROOT,
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BATCH_SIZE,
)
from src.model.weather_net import WeatherNet, count_parameters
from src.data.preprocessing import load_training_data


console = Console()


class WeatherLoss(nn.Module):
    """Combined loss for weather prediction.
    
    Uses MSE for temperature and wind speed, BCE for precipitation probability.
    """
    
    def __init__(self, temp_weight: float = 1.0, precip_weight: float = 0.5, wind_weight: float = 0.5):
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
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.
        
        Args:
            predictions: Model predictions, shape (batch, seq_len, 3)
                         Features: [temp, precip, wind]
            targets: Ground truth, shape (batch, seq_len, 3)
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Extract individual predictions (indices: 0=temp, 1=precip, 2=wind)
        pred_temp = predictions[:, :, 0]
        pred_precip = predictions[:, :, 1]  # Raw logits for BCE
        pred_wind = predictions[:, :, 2]
        
        target_temp = targets[:, :, 0]
        target_precip = targets[:, :, 1]  # Already normalized 0-1
        target_wind = targets[:, :, 2]
        
        # Compute individual losses
        temp_loss = self.mse(pred_temp, target_temp)
        # For BCE, we need to clamp target to valid range [0, 1]
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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        teacher_forcing_ratio: Probability of using teacher forcing
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    total_losses = {"temp_loss": 0, "precip_loss": 0, "wind_loss": 0, "total_loss": 0}
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        predictions = model(batch_x, teacher_forcing_ratio=teacher_forcing_ratio, target=batch_y)
        
        # Compute loss
        loss, loss_dict = criterion(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1
    
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
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def train_model(
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    teacher_forcing_start: float = 0.5,
    teacher_forcing_decay: float = 0.95,
    model_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the WeatherNet model.
    
    Args:
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        patience: Early stopping patience (epochs without improvement)
        batch_size: Batch size for training
        teacher_forcing_start: Initial teacher forcing ratio
        teacher_forcing_decay: Decay rate for teacher forcing per epoch
        model_path: Path to save best model (default: models/best_model.pt)
        log_path: Path to save training log (default: training_log.csv)
    
    Returns:
        model: Trained model
        history: Training history dictionary
    """
    # Set up paths
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pt"
    if log_path is None:
        log_path = PROJECT_ROOT / "training_log.csv"
    
    # Ensure directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"\n[bold cyan]Weather Oracle - Training Pipeline[/bold cyan]")
    console.print(f"Device: {device}")
    
    # Load data
    console.print("\nLoading data...")
    train_loader, val_loader, test_loader = load_training_data(batch_size=batch_size)
    
    # Create model
    model = WeatherNet().to(device)
    console.print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function
    criterion = WeatherLoss()
    
    # Optimizer with Adam
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    
    # Training state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    teacher_forcing = teacher_forcing_start
    
    # History
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_temp_loss": [],
        "val_temp_loss": [],
        "train_precip_loss": [],
        "val_precip_loss": [],
        "train_wind_loss": [],
        "val_wind_loss": [],
        "lr": [],
    }
    
    # Initialize CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "train_temp_loss", "val_temp_loss",
            "train_precip_loss", "val_precip_loss", "train_wind_loss", "val_wind_loss",
            "lr", "best_val_loss", "teacher_forcing"
        ])
    
    console.print(f"\nTraining for up to {epochs} epochs...")
    console.print(f"Early stopping patience: {patience}")
    console.print(f"Model will be saved to: {model_path}")
    console.print(f"Training log: {log_path}\n")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, teacher_forcing
        )
        
        # Validate
        val_losses = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_losses["total_loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Record history
        history["train_loss"].append(train_losses["total_loss"])
        history["val_loss"].append(val_losses["total_loss"])
        history["train_temp_loss"].append(train_losses["temp_loss"])
        history["val_temp_loss"].append(val_losses["temp_loss"])
        history["train_precip_loss"].append(train_losses["precip_loss"])
        history["val_precip_loss"].append(val_losses["precip_loss"])
        history["train_wind_loss"].append(train_losses["wind_loss"])
        history["val_wind_loss"].append(val_losses["wind_loss"])
        history["lr"].append(current_lr)
        
        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_losses["total_loss"], val_losses["total_loss"],
                train_losses["temp_loss"], val_losses["temp_loss"],
                train_losses["precip_loss"], val_losses["precip_loss"],
                train_losses["wind_loss"], val_losses["wind_loss"],
                current_lr, best_val_loss, teacher_forcing
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
                "val_loss": best_val_loss,
                "history": history,
            }, model_path)
            
            improvement_str = "[green]✓ Best[/green]"
        else:
            epochs_without_improvement += 1
            improvement_str = ""
        
        # Print epoch progress
        epoch_time = time.time() - epoch_start
        console.print(
            f"🧠 Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_losses['total_loss']:.4f} | "
            f"Val Loss: {val_losses['total_loss']:.4f} | "
            f"Best: {best_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"TF: {teacher_forcing:.2f} | "
            f"{epoch_time:.1f}s {improvement_str}"
        )
        
        # Decay teacher forcing
        teacher_forcing = max(0.0, teacher_forcing * teacher_forcing_decay)
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            console.print(f"\n[yellow]Early stopping triggered after {epoch} epochs[/yellow]")
            break
    
    total_time = time.time() - start_time
    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Total time: {total_time / 60:.1f} minutes")
    console.print(f"Best validation loss: {best_val_loss:.4f}")
    console.print(f"Model saved to: {model_path}")
    
    # Load best model for return
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, history


def main():
    """Main entry point for training."""
    import sys
    
    # Default parameters
    epochs = EPOCHS
    lr = LEARNING_RATE
    patience = EARLY_STOPPING_PATIENCE
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            console.print("[red]Usage: python -m src.training.trainer [epochs][/red]")
            sys.exit(1)
    
    train_model(epochs=epochs, lr=lr, patience=patience)


if __name__ == "__main__":
    main()
