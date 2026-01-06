"""Configuration settings for Weather Oracle."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Database
DB_PATH = DATA_DIR / "weather.db"

# Open-Meteo API URLs (no API key needed)
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Rate limiting
API_RATE_LIMIT_SECONDS = 1.0  # Max 1 request per second

# Model hyperparameters
SEQUENCE_INPUT_HOURS = 72   # 3 days of input
SEQUENCE_OUTPUT_HOURS = 24  # 1 day of predictions
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Data splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Weather features
INPUT_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
]

OUTPUT_FEATURES = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]
