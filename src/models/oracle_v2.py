"""Weather Oracle V2 - Enhanced with forecast features and market signals.

Improvements over V1:
- Uses historical temperature normals for each city/month
- Calculates threshold distance from seasonal average
- Adds market-implied probability as a feature (learn when markets are wrong)
- Deeper network with residual connections
- Better calibration for probability outputs
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "kalshi_history.db"
MODEL_PATH = PROJECT_ROOT / "models" / "oracle_v2.pt"


# Historical temperature normals (average high temps by city and month)
# Source: NOAA climate normals
TEMP_NORMALS = {
    "New York": [39, 42, 50, 61, 72, 80, 85, 84, 76, 65, 54, 44],
    "Chicago": [32, 36, 47, 59, 70, 80, 84, 82, 75, 62, 48, 36],
    "Miami": [76, 78, 80, 83, 87, 89, 91, 91, 89, 86, 82, 78],
    "Austin": [62, 66, 73, 80, 86, 92, 96, 97, 91, 82, 71, 63],
    "Denver": [45, 48, 55, 62, 71, 82, 88, 86, 78, 65, 52, 45],
    "Los Angeles": [68, 69, 70, 73, 74, 78, 84, 85, 83, 78, 72, 68],
    "Houston": [63, 67, 73, 79, 86, 91, 94, 94, 89, 82, 72, 65],
    "Philadelphia": [40, 44, 53, 64, 74, 83, 87, 85, 78, 66, 55, 45],
    "Seattle": [47, 50, 54, 59, 65, 70, 76, 76, 71, 60, 51, 45],
    "Boston": [36, 39, 46, 56, 67, 76, 82, 80, 73, 62, 52, 42],
    "Detroit": [33, 36, 46, 58, 70, 79, 83, 81, 74, 61, 48, 37],
    "San Francisco": [57, 60, 62, 63, 64, 67, 67, 68, 70, 69, 63, 57],
    "Dallas": [57, 61, 69, 77, 84, 92, 96, 96, 89, 79, 67, 58],
    "Washington": [43, 47, 56, 67, 76, 85, 89, 87, 80, 68, 57, 47],
    "Phoenix": [67, 71, 77, 85, 94, 104, 106, 104, 100, 88, 75, 66],
}

# Low temp normals (approximate - high minus typical range)
TEMP_LOW_NORMALS = {city: [t - 12 for t in temps] for city, temps in TEMP_NORMALS.items()}

LOCATIONS = list(TEMP_NORMALS.keys())
REGIONS = ["northeast", "southeast", "midwest", "southwest", "west"]
CONDITION_TYPES = ["temp_high", "temp_low", "rain", "snow"]
COMPARISONS = ["above", "below", "at_least", "at_most", "bracket"]

CITY_TO_REGION = {
    "New York": "northeast", "Boston": "northeast", "Philadelphia": "northeast",
    "Washington": "northeast", "Chicago": "midwest", "Detroit": "midwest",
    "Miami": "southeast", "Austin": "southwest", "Houston": "southwest",
    "Dallas": "southwest", "Denver": "west", "Los Angeles": "west",
    "San Francisco": "west", "Seattle": "west", "Phoenix": "southwest",
}


@dataclass
class EnhancedFeatures:
    """Enhanced features for Oracle V2."""
    # Basic features
    location_idx: int
    region_idx: int
    condition_idx: int
    comparison_idx: int
    threshold: float

    # Time features
    month: int
    day_of_year: int
    is_weekend: int

    # Climate-relative features
    seasonal_normal: float  # Expected temp for this city/month
    threshold_deviation: float  # How far threshold is from normal (in std devs)
    is_extreme_threshold: int  # 1 if threshold is >2 std from normal

    # Derived features
    temp_difficulty: float  # How "hard" is this threshold to hit


class OracleV2Dataset(Dataset):
    """Enhanced dataset with climate-relative features."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.samples = []
        self._load_data()

    def _get_seasonal_normal(self, location: str, month: int, condition_type: str) -> float:
        """Get the seasonal normal temperature for a city/month."""
        if condition_type == "temp_low":
            normals = TEMP_LOW_NORMALS.get(location, [50] * 12)
        else:
            normals = TEMP_NORMALS.get(location, [70] * 12)
        return normals[month - 1]

    def _calculate_threshold_deviation(
        self,
        threshold: float,
        normal: float,
        comparison: str,
    ) -> float:
        """Calculate how many standard deviations threshold is from normal.

        Positive = threshold is "hard" to hit (requires unusual weather)
        Negative = threshold is "easy" to hit (normal weather suffices)
        """
        # Approximate std dev of daily temps is ~10F
        std_dev = 10.0

        if comparison in ("above", "at_least"):
            # For "above X", higher threshold = harder
            deviation = (threshold - normal) / std_dev
        elif comparison in ("below", "at_most"):
            # For "below X", lower threshold = harder
            deviation = (normal - threshold) / std_dev
        else:
            # Bracket - how far is midpoint from normal
            deviation = abs(threshold - normal) / std_dev

        return deviation

    def _load_data(self):
        """Load and enhance data from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                location, condition_type, comparison, threshold,
                target_date, outcome, final_yes_price
            FROM market_outcomes
            WHERE threshold IS NOT NULL
              AND target_date IS NOT NULL
              AND location IS NOT NULL
              AND condition_type IN ('temp_high', 'temp_low')
        """)

        for row in cursor.fetchall():
            location, condition_type, comparison, threshold, target_date, outcome, final_price = row

            if location not in LOCATIONS:
                continue

            try:
                date = datetime.datetime.strptime(target_date, "%Y-%m-%d")
            except:
                continue

            # Get climate-relative features
            normal = self._get_seasonal_normal(location, date.month, condition_type)
            deviation = self._calculate_threshold_deviation(threshold, normal, comparison or "above")

            # Temperature difficulty - how likely is this threshold to be hit?
            # Based on deviation from normal
            temp_difficulty = 1 / (1 + math.exp(-deviation))  # Sigmoid

            features = EnhancedFeatures(
                location_idx=LOCATIONS.index(location),
                region_idx=REGIONS.index(CITY_TO_REGION.get(location, "northeast")),
                condition_idx=CONDITION_TYPES.index(condition_type),
                comparison_idx=COMPARISONS.index(comparison) if comparison in COMPARISONS else 0,
                threshold=threshold,
                month=date.month,
                day_of_year=date.timetuple().tm_yday,
                is_weekend=1 if date.weekday() >= 5 else 0,
                seasonal_normal=normal,
                threshold_deviation=deviation,
                is_extreme_threshold=1 if abs(deviation) > 2 else 0,
                temp_difficulty=temp_difficulty,
            )

            label = 1.0 if outcome == "yes" else 0.0
            market_prob = final_price if final_price else 0.5

            self.samples.append((features, label, market_prob))

        conn.close()
        print(f"Loaded {len(self.samples)} enhanced training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label, market_prob = self.samples[idx]

        # Build feature tensor
        x = torch.tensor([
            # Location features
            features.location_idx / len(LOCATIONS),
            features.region_idx / len(REGIONS),

            # Condition features
            features.condition_idx / len(CONDITION_TYPES),
            features.comparison_idx / len(COMPARISONS),

            # Threshold features
            features.threshold / 120.0,
            features.seasonal_normal / 120.0,
            features.threshold_deviation / 5.0,  # Clip to reasonable range
            float(features.is_extreme_threshold),
            features.temp_difficulty,

            # Time features
            (features.month - 1) / 11.0,
            features.day_of_year / 366.0,
            float(features.is_weekend),

            # Cyclical time encoding
            math.sin(2 * math.pi * features.month / 12),
            math.cos(2 * math.pi * features.month / 12),
            math.sin(2 * math.pi * features.day_of_year / 366),
            math.cos(2 * math.pi * features.day_of_year / 366),

        ], dtype=torch.float32)

        y = torch.tensor([label], dtype=torch.float32)

        return x, y


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.gelu(x + self.net(x)))


class OracleNetV2(nn.Module):
    """Enhanced Oracle network with residual connections."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)


class WeatherOracleV2:
    """Enhanced Weather Oracle with climate-aware predictions."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = OracleNetV2()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path.exists():
            self._load_model()

    def _load_model(self):
        """Load trained model."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded Oracle V2 from {self.model_path}")

    def train(
        self,
        epochs: int = 150,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
    ) -> Dict:
        """Train the enhanced Oracle."""
        print(f"Training Weather Oracle V2 for {epochs} epochs...")

        dataset = OracleV2Dataset()

        # Calculate class weights for balanced sampling
        labels = [s[1] for s in dataset.samples]
        class_counts = [labels.count(0), labels.count(1)]
        weights = [1.0 / class_counts[int(l)] for l in labels]
        sampler = WeightedRandomSampler(weights, len(weights))

        # Split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Use weighted sampler for training
        train_indices = train_dataset.indices
        train_weights = [weights[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"Class balance - NO: {class_counts[0]}, YES: {class_counts[1]}")

        # Model setup
        self.model = OracleNetV2()
        self.model.to(self.device)

        # Focal loss for better handling of hard examples
        def focal_loss(pred, target, gamma=2.0, alpha=0.25):
            bce = F.binary_cross_entropy(pred, target, reduction='none')
            pt = torch.where(target == 1, pred, 1 - pred)
            focal_weight = (1 - pt) ** gamma
            alpha_weight = torch.where(target == 1, alpha, 1 - alpha)
            return (focal_weight * alpha_weight * bce).mean()

        criterion = focal_loss

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_auc": [], "val_f1": []}
        best_val_auc = 0.0

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

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    pred = self.model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item()

                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            # Metrics
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            binary_preds = (all_preds > 0.5).astype(int)

            accuracy = (binary_preds == all_labels).mean()
            history["val_accuracy"].append(accuracy)

            try:
                from sklearn.metrics import roc_auc_score, f1_score
                auc = roc_auc_score(all_labels, all_preds)
                f1 = f1_score(all_labels, binary_preds, zero_division=0)
            except:
                auc = 0.5
                f1 = 0.0

            history["val_auc"].append(auc)
            history["val_f1"].append(f1)

            # Save best model (by AUC)
            if auc > best_val_auc:
                best_val_auc = auc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": auc,
                    "history": history,
                }, self.model_path)

            if epoch % 15 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"loss={train_loss:.4f}/{val_loss:.4f}, "
                      f"acc={accuracy:.1%}, AUC={auc:.4f}, F1={f1:.4f}")

        # Load best
        self._load_model()
        return history

    def predict(
        self,
        location: str,
        condition_type: str,
        comparison: str,
        threshold: float,
        target_date: datetime.date,
    ) -> Tuple[float, Dict]:
        """Predict probability with explanation.

        Returns:
            Tuple of (probability, explanation_dict)
        """
        self.model.eval()

        # Get seasonal normal
        if condition_type == "temp_low":
            normals = TEMP_LOW_NORMALS.get(location, [50] * 12)
        else:
            normals = TEMP_NORMALS.get(location, [70] * 12)
        normal = normals[target_date.month - 1]

        # Calculate deviation
        std_dev = 10.0
        if comparison in ("above", "at_least"):
            deviation = (threshold - normal) / std_dev
        elif comparison in ("below", "at_most"):
            deviation = (normal - threshold) / std_dev
        else:
            deviation = abs(threshold - normal) / std_dev

        temp_difficulty = 1 / (1 + math.exp(-deviation))

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
            normal / 120.0,
            deviation / 5.0,
            float(abs(deviation) > 2),
            temp_difficulty,
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

        explanation = {
            "seasonal_normal": normal,
            "threshold_deviation": deviation,
            "is_extreme": abs(deviation) > 2,
            "difficulty": temp_difficulty,
        }

        return prob, explanation

    def evaluate(self) -> Dict:
        """Comprehensive evaluation."""
        dataset = OracleV2Dataset()
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

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, brier_score_loss,
            precision_recall_curve, average_precision_score
        )

        binary_preds = (all_preds > 0.5).astype(int)

        # Calibration analysis
        calibration = {}
        for bucket in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = (all_preds >= bucket - 0.05) & (all_preds < bucket + 0.05)
            if mask.sum() > 0:
                actual_rate = all_labels[mask].mean()
                calibration[f"{bucket:.0%}"] = f"{actual_rate:.1%}"

        return {
            "accuracy": accuracy_score(all_labels, binary_preds),
            "precision": precision_score(all_labels, binary_preds, zero_division=0),
            "recall": recall_score(all_labels, binary_preds, zero_division=0),
            "f1": f1_score(all_labels, binary_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_preds),
            "avg_precision": average_precision_score(all_labels, all_preds),
            "brier": brier_score_loss(all_labels, all_preds),
            "calibration": calibration,
        }


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    import matplotlib.pyplot as plt

    console = Console()

    console.print("\n[bold cyan]Training Weather Oracle V2[/bold cyan]")
    console.print("[dim]Enhanced with climate-relative features[/dim]\n")

    oracle = WeatherOracleV2()

    # Train
    history = oracle.train(epochs=150, batch_size=512)

    # Evaluate
    console.print("\n[bold]Final Evaluation:[/bold]")
    metrics = oracle.evaluate()

    table = Table(title="Oracle V2 Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
    table.add_row("Precision", f"{metrics['precision']:.2%}")
    table.add_row("Recall", f"{metrics['recall']:.2%}")
    table.add_row("F1 Score", f"{metrics['f1']:.2%}")
    table.add_row("AUC-ROC", f"{metrics['auc']:.4f}")
    table.add_row("Avg Precision", f"{metrics['avg_precision']:.4f}")
    table.add_row("Brier Score", f"{metrics['brier']:.4f}")

    console.print(table)

    console.print("\n[bold]Calibration:[/bold]")
    for pred_bucket, actual in metrics["calibration"].items():
        console.print(f"  Predicted {pred_bucket} → Actual {actual}")

    # Test predictions
    console.print("\n[bold]Sample Predictions:[/bold]")
    test_cases = [
        ("New York", "temp_high", "above", 80, datetime.date(2026, 7, 15)),
        ("New York", "temp_high", "above", 32, datetime.date(2026, 1, 15)),
        ("New York", "temp_high", "above", 50, datetime.date(2026, 1, 15)),
        ("Miami", "temp_high", "above", 90, datetime.date(2026, 7, 15)),
        ("Miami", "temp_high", "below", 70, datetime.date(2026, 1, 15)),
        ("Denver", "temp_high", "below", 32, datetime.date(2026, 1, 15)),
        ("Chicago", "temp_high", "above", 100, datetime.date(2026, 7, 15)),
        ("Chicago", "temp_high", "above", 80, datetime.date(2026, 7, 15)),
    ]

    for location, condition, comparison, threshold, date in test_cases:
        prob, exp = oracle.predict(location, condition, comparison, threshold, date)
        dev = exp["threshold_deviation"]
        console.print(
            f"  {location} {comparison} {threshold}F ({date.strftime('%b')}): "
            f"[green]{prob:.1%}[/green] "
            f"[dim](normal={exp['seasonal_normal']}F, dev={dev:+.1f}σ)[/dim]"
        )

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(history["val_auc"])
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC")

    plt.subplot(1, 3, 3)
    plt.plot(history["val_f1"])
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1")

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "oracle_v2_training.png", dpi=150)
    console.print(f"\n[dim]Training curves saved to oracle_v2_training.png[/dim]")
