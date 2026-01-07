"""Configuration settings for Weather Oracle."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")
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

# Kalshi API credentials
KALSHI_CLIENT_ID = os.getenv("KALSHI_CLIENT_ID", "")
KALSHI_CLIENT_SECRET = os.getenv("KALSHI_CLIENT_SECRET", "")
KALSHI_API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Telegram Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
