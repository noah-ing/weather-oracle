"""Weather Oracle Model - Learns from actual Kalshi market outcomes.

This is the TRUE Oracle - it doesn't synthesize forecasts, it learns what
actually happens. Trained on 37K+ real market outcomes from Kalshi.

Key Innovation:
- Inputs: Market characteristics (location, threshold, condition type, time)
- Output: P(event happens) - the actual probability
- Training: Historical outcomes (did the event actually occur?)

This model learns patterns like:
- "NYC high temp >80F in January rarely happens" (regardless of forecasts)
- "Miami rain markets have higher YES rates than Denver"
- "Extreme thresholds (>100F) almost never hit"
"""

import datetime
import sqlite3
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import PROJECT_ROOT


# Database path
DB_PATH = PROJECT_ROOT / "data" / "kalshi_history.db"
MODEL_PATH = PROJECT_ROOT / "models" / "oracle.pt"


# City to region mapping
CITY_TO_REGION = {
    "New York": "northeast",
    "Boston": "northeast",
    "Philadelphia": "northeast",
    "Washington": "northeast",
    "Chicago": "midwest",
    "Detroit": "midwest",
    "Miami": "southeast",
    "Austin": "southwest",
    "Houston": "southwest",
    "Dallas": "southwest",
    "Denver": "west",
    "Los Angeles": "west",
    "San Francisco": "west",
    "Seattle": "west",
    "Phoenix": "southwest",
}

# Encode categorical features
LOCATIONS = list(CITY_TO_REGION.keys())
REGIONS = ["northeast", "southeast", "midwest", "southwest", "west"]
CONDITION_TYPES = ["temp_high", "temp_low", "rain", "snow"]
COMPARISONS = ["above", "below", "at_least", "at_most", "bracket"]


@dataclass
class MarketFeatures:
    """Features extracted from a market for prediction."""
    location_idx: int
    region_idx: int
    condition_idx: int
    comparison_idx: int
    threshold: float
    month: int  # 1-12
    day_of_year: int  # 1-366
    is_weekend: int  # 0 or 1


class OracleDataset(Dataset):
    """Dataset of historical market outcomes for training."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load data from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                location, condition_type, comparison, threshold,
                target_date, outcome
            FROM market_outcomes
            WHERE threshold IS NOT NULL
              AND target_date IS NOT NULL
              AND location IS NOT NULL
        """)

        for row in cursor.fetchall():
            location, condition_type, comparison, threshold, target_date, outcome = row

            # Skip unknown locations
            if location not in LOCATIONS:
                continue

            # Parse date
            try:
                date = datetime.datetime.strptime(target_date, "%Y-%m-%d")
            except:
                continue

            # Encode features
            features = MarketFeatures(
                location_idx=LOCATIONS.index(location),
                region_idx=REGIONS.index(CITY_TO_REGION.get(location, "northeast")),
                condition_idx=CONDITION_TYPES.index(condition_type) if condition_type in CONDITION_TYPES else 0,
                comparison_idx=COMPARISONS.index(comparison) if comparison in COMPARISONS else 0,
                threshold=threshold,
                month=date.month,
                day_of_year=date.timetuple().tm_yday,
                is_weekend=1 if date.weekday() >= 5 else 0,
            )

            # Label: 1 if YES, 0 if NO
            label = 1.0 if outcome == "yes" else 0.0

            self.samples.append((features, label))

        conn.close()
        print(f"Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]

        # Convert to tensor
        x = torch.tensor([
            features.location_idx / len(LOCATIONS),  # Normalized
            features.region_idx / len(REGIONS),
            features.condition_idx / len(CONDITION_TYPES),
            features.comparison_idx / len(COMPARISONS),
            features.threshold / 120.0,  # Normalize temp (assume max 120F)
            (features.month - 1) / 11.0,  # 0-1
            features.day_of_year / 366.0,  # 0-1
            float(features.is_weekend),
            # Cyclical encoding for month
            math.sin(2 * math.pi * features.month / 12),
            math.cos(2 * math.pi * features.month / 12),
            # Cyclical encoding for day of year
            math.sin(2 * math.pi * features.day_of_year / 366),
            math.cos(2 * math.pi * features.day_of_year / 366),
        ], dtype=torch.float32)

        y = torch.tensor([label], dtype=torch.float32)

        return x, y


class OracleNet(nn.Module):
    """Neural network for predicting market outcomes.

    Simple but effective architecture:
    - Input: 12 features (location, condition, threshold, time)
    - Hidden layers with dropout for regularization
    - Output: Probability that YES outcome occurs
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class WeatherOracle:
    """The Weather Oracle - predicts actual event outcomes."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = OracleNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path.exists():
            self._load_model()

    def _load_model(self):
        """Load trained model."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded Oracle model from {self.model_path}")

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
    ) -> Dict:
        """Train the Oracle on historical outcomes.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio

        Returns:
            Training history
        """
        print(f"Training Weather Oracle for {epochs} epochs...")

        # Load dataset
        dataset = OracleDataset()

        # Split into train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Model setup
        self.model = OracleNet()
        self.model.to(self.device)

        # Handle class imbalance - YES is ~18%, NO is ~82%
        # Use weighted BCE loss to penalize missing YES more heavily
        def weighted_bce_loss(pred, target):
            # Weight YES (1) samples more heavily
            weight = torch.where(target == 1, 4.5, 1.0)  # 82/18 ≈ 4.5
            bce = F.binary_cross_entropy(pred, target, reduction='none')
            return (bce * weight).mean()

        criterion = weighted_bce_loss

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_auc": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    pred = self.model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item()

                    # Accuracy
                    predicted = (pred > 0.5).float()
                    correct += (predicted == y).sum().item()
                    total += y.size(0)

                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            val_loss /= len(val_loader)
            accuracy = correct / total
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(accuracy)

            # Calculate AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(all_labels, all_preds)
                history["val_auc"].append(auc)
            except:
                auc = 0.0
                history["val_auc"].append(auc)

            scheduler.step()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "history": history,
                }, self.model_path)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"accuracy={accuracy:.2%}, "
                      f"AUC={auc:.4f}")

        # Load best model
        self._load_model()

        return history

    def predict(
        self,
        location: str,
        condition_type: str,
        comparison: str,
        threshold: float,
        target_date: datetime.date,
    ) -> float:
        """Predict probability of event occurring.

        Args:
            location: City name
            condition_type: temp_high, temp_low, rain, snow
            comparison: above, below, at_least, at_most, bracket
            threshold: Temperature or precipitation threshold
            target_date: Date of the event

        Returns:
            Probability (0-1) that the event occurs (YES wins)
        """
        self.model.eval()

        # Build features
        location_idx = LOCATIONS.index(location) if location in LOCATIONS else 0
        region_idx = REGIONS.index(CITY_TO_REGION.get(location, "northeast"))
        condition_idx = CONDITION_TYPES.index(condition_type) if condition_type in CONDITION_TYPES else 0
        comparison_idx = COMPARISONS.index(comparison) if comparison in COMPARISONS else 0

        x = torch.tensor([[
            location_idx / len(LOCATIONS),
            region_idx / len(REGIONS),
            condition_idx / len(CONDITION_TYPES),
            comparison_idx / len(COMPARISONS),
            threshold / 120.0,
            (target_date.month - 1) / 11.0,
            target_date.timetuple().tm_yday / 366.0,
            float(target_date.weekday() >= 5),
            math.sin(2 * math.pi * target_date.month / 12),
            math.cos(2 * math.pi * target_date.month / 12),
            math.sin(2 * math.pi * target_date.timetuple().tm_yday / 366),
            math.cos(2 * math.pi * target_date.timetuple().tm_yday / 366),
        ]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prob = self.model(x).item()

        return prob

    def evaluate(self) -> Dict:
        """Evaluate model on full dataset."""
        dataset = OracleDataset()
        loader = DataLoader(dataset, batch_size=256)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                pred = self.model(x)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, brier_score_loss
        )

        binary_preds = (all_preds > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(all_labels, binary_preds),
            "precision": precision_score(all_labels, binary_preds, zero_division=0),
            "recall": recall_score(all_labels, binary_preds, zero_division=0),
            "f1": f1_score(all_labels, binary_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_preds),
            "brier": brier_score_loss(all_labels, all_preds),
        }


def train_oracle(epochs: int = 100) -> Dict:
    """Convenience function to train the Oracle."""
    oracle = WeatherOracle()
    return oracle.train(epochs=epochs)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    import matplotlib.pyplot as plt

    console = Console()

    console.print("\n[bold cyan]Training Weather Oracle[/bold cyan]")
    console.print("[dim]Learning from 37K+ real Kalshi outcomes[/dim]\n")

    oracle = WeatherOracle()

    # Train
    history = oracle.train(epochs=100, batch_size=512)

    # Evaluate
    console.print("\n[bold]Final Evaluation:[/bold]")
    metrics = oracle.evaluate()

    table = Table(title="Oracle Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
    table.add_row("Precision", f"{metrics['precision']:.2%}")
    table.add_row("Recall", f"{metrics['recall']:.2%}")
    table.add_row("F1 Score", f"{metrics['f1']:.2%}")
    table.add_row("AUC-ROC", f"{metrics['auc']:.4f}")
    table.add_row("Brier Score", f"{metrics['brier']:.4f}")

    console.print(table)

    # Test predictions
    console.print("\n[bold]Sample Predictions:[/bold]")
    test_cases = [
        ("New York", "temp_high", "above", 80, datetime.date(2026, 7, 15)),
        ("New York", "temp_high", "above", 32, datetime.date(2026, 1, 15)),
        ("Miami", "temp_high", "above", 80, datetime.date(2026, 1, 15)),
        ("Denver", "temp_high", "below", 32, datetime.date(2026, 1, 15)),
        ("Chicago", "temp_high", "above", 100, datetime.date(2026, 7, 15)),
    ]

    for location, condition, comparison, threshold, date in test_cases:
        prob = oracle.predict(location, condition, comparison, threshold, date)
        console.print(f"  {location} {condition} {comparison} {threshold}F on {date}: [green]{prob:.1%}[/green]")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(history["val_accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(history["val_auc"])
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC")

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "oracle_training.png", dpi=150)
    console.print(f"\n[dim]Training curves saved to oracle_training.png[/dim]")
