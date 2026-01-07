"""Data preprocessing pipeline for weather forecasting model."""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    SEQUENCE_INPUT_HOURS,
    SEQUENCE_OUTPUT_HOURS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.db.database import get_all_observations, init_db


# Feature columns in the database (input features)
DB_INPUT_FEATURES = ["temp", "humidity", "wind_speed", "precipitation", "pressure", "cloud_cover"]

# Output features we want to predict
DB_OUTPUT_FEATURES = ["temp", "precipitation", "wind_speed"]


class WeatherDataset(Dataset):
    """PyTorch Dataset for weather sequences."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """Initialize dataset with sequences and targets.

        Args:
            sequences: Input sequences of shape (N, input_hours, num_features)
            targets: Target sequences of shape (N, output_hours, num_targets)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def load_data_from_db() -> pd.DataFrame:
    """Load all observation data from SQLite database.

    Returns:
        DataFrame with weather observations sorted by city and timestamp
    """
    init_db()  # Ensure database exists
    observations = get_all_observations()

    if not observations:
        raise ValueError("No observations found in database. Run data collection first.")

    df = pd.DataFrame(observations)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by city and timestamp for proper sequence creation
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    return df


def create_sequences(
    df: pd.DataFrame,
    input_features: list[str],
    output_features: list[str],
    input_hours: int = SEQUENCE_INPUT_HOURS,
    output_hours: int = SEQUENCE_OUTPUT_HOURS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create input/output sequences for time series forecasting.

    Args:
        df: DataFrame with weather data
        input_features: Column names for input features
        output_features: Column names for output targets
        input_hours: Number of hours for input sequence (default 72)
        output_hours: Number of hours for output sequence (default 24)

    Returns:
        Tuple of (input_sequences, output_sequences)
        - input_sequences: shape (N, input_hours, len(input_features))
        - output_sequences: shape (N, output_hours, len(output_features))
    """
    sequences = []
    targets = []

    sequence_length = input_hours + output_hours

    # Process each city separately to avoid cross-city sequences
    for city in df["city"].unique():
        city_data = df[df["city"] == city].copy()

        # Check for continuous hourly data
        city_data = city_data.sort_values("timestamp").reset_index(drop=True)

        # Extract feature values
        input_values = city_data[input_features].values
        output_values = city_data[output_features].values

        # Skip rows with NaN values
        valid_mask = ~np.isnan(input_values).any(axis=1) & ~np.isnan(output_values).any(axis=1)

        # Create sliding window sequences
        for i in range(len(city_data) - sequence_length + 1):
            # Check if all rows in this window are valid
            window_valid = valid_mask[i : i + sequence_length].all()
            if not window_valid:
                continue

            # Also check for temporal continuity (hourly gaps)
            timestamps = city_data["timestamp"].iloc[i : i + sequence_length]
            time_diffs = timestamps.diff().iloc[1:]  # Skip first NaT
            max_gap = time_diffs.max()

            # Only accept sequences with <= 1 hour gaps (allow for some flexibility)
            if pd.notna(max_gap) and max_gap <= pd.Timedelta(hours=1):
                seq = input_values[i : i + input_hours]
                target = output_values[i + input_hours : i + sequence_length]
                sequences.append(seq)
                targets.append(target)

    if not sequences:
        raise ValueError("No valid sequences created. Check data continuity and quality.")

    return np.array(sequences), np.array(targets)


def normalize_data(
    sequences: np.ndarray,
    targets: np.ndarray,
    scaler_path: Path = DATA_DIR / "scaler.pkl",
    fit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Normalize input sequences and targets using MinMaxScaler.

    Args:
        sequences: Input sequences of shape (N, input_hours, num_features)
        targets: Target sequences of shape (N, output_hours, num_targets)
        scaler_path: Path to save/load scaler parameters
        fit: If True, fit the scaler; if False, load existing scaler

    Returns:
        Tuple of (normalized_sequences, normalized_targets, scaler_info)
    """
    if fit:
        # Reshape for fitting: (N * hours, features)
        n_samples, input_hours, n_input_features = sequences.shape
        _, output_hours, n_output_features = targets.shape

        # Create separate scalers for input and output features
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        output_scaler = MinMaxScaler(feature_range=(0, 1))

        # Reshape to 2D for fitting
        sequences_2d = sequences.reshape(-1, n_input_features)
        targets_2d = targets.reshape(-1, n_output_features)

        # Fit and transform
        sequences_normalized = input_scaler.fit_transform(sequences_2d)
        targets_normalized = output_scaler.fit_transform(targets_2d)

        # Reshape back to 3D
        sequences_normalized = sequences_normalized.reshape(n_samples, input_hours, n_input_features)
        targets_normalized = targets_normalized.reshape(n_samples, output_hours, n_output_features)

        # Save scaler info
        scaler_info = {
            "input_scaler": input_scaler,
            "output_scaler": output_scaler,
            "input_features": DB_INPUT_FEATURES,
            "output_features": DB_OUTPUT_FEATURES,
        }

        # Ensure data directory exists
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_info, f)

        print(f"Scaler saved to {scaler_path}")

    else:
        # Load existing scaler
        with open(scaler_path, "rb") as f:
            scaler_info = pickle.load(f)

        input_scaler = scaler_info["input_scaler"]
        output_scaler = scaler_info["output_scaler"]

        # Transform
        n_samples, input_hours, n_input_features = sequences.shape
        _, output_hours, n_output_features = targets.shape

        sequences_2d = sequences.reshape(-1, n_input_features)
        targets_2d = targets.reshape(-1, n_output_features)

        sequences_normalized = input_scaler.transform(sequences_2d)
        targets_normalized = output_scaler.transform(targets_2d)

        sequences_normalized = sequences_normalized.reshape(n_samples, input_hours, n_input_features)
        targets_normalized = targets_normalized.reshape(n_samples, output_hours, n_output_features)

    return sequences_normalized, targets_normalized, scaler_info


def load_training_data(
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare training data from SQLite database.

    Args:
        batch_size: Batch size for DataLoaders (default from config)
        train_split: Fraction of data for training (default 0.8)
        val_split: Fraction of data for validation (default 0.1)
        test_split: Fraction of data for testing (default 0.1)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("Loading data from database...")
    df = load_data_from_db()
    print(f"Loaded {len(df):,} observations from {df['city'].nunique()} cities")

    print("Creating sequences (72h input -> 24h output)...")
    sequences, targets = create_sequences(df, DB_INPUT_FEATURES, DB_OUTPUT_FEATURES)
    print(f"Created {len(sequences):,} sequences")
    print(f"Input shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")

    print("Normalizing data...")
    sequences_norm, targets_norm, scaler_info = normalize_data(sequences, targets)

    # Create full dataset
    dataset = WeatherDataset(sequences_norm, targets_norm)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    print(f"Splitting data: {train_size:,} train, {val_size:,} val, {test_size:,} test")

    # Random split with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"DataLoaders created with batch_size={batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def load_scaler(scaler_path: Path = DATA_DIR / "scaler.pkl") -> dict:
    """Load saved scaler from disk.

    Args:
        scaler_path: Path to saved scaler pickle file

    Returns:
        Dictionary containing input_scaler, output_scaler, and feature names
    """
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Run preprocessing when executed as script
    train_loader, val_loader, test_loader = load_training_data()

    # Print sample batch shapes
    for batch_x, batch_y in train_loader:
        print(f"\nSample batch:")
        print(f"  Input shape: {batch_x.shape}")
        print(f"  Target shape: {batch_y.shape}")
        break
