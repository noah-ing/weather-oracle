# Weather Oracle

A neural network weather forecasting system with Kalshi prediction market integration. Weather Oracle combines multiple weather data sources, LSTM-based neural networks, and probabilistic calibration to identify edge opportunities in weather prediction markets.

## Features

### Core Forecasting
- **LSTM Encoder-Decoder**: Neural network with attention mechanism for 24-hour weather predictions
- **Multi-Model Ensemble**: Combines forecasts from GFS, ICON, GEM, NWS, and trained NN model
- **Bias Correction**: Per-source, per-location bias tracking and automatic correction
- **Probabilistic Calibration**: Gaussian CDF-based probability estimation for threshold questions

### Kalshi Integration
- **Market Scanner**: Finds active weather prediction markets with parsed location, date, and threshold
- **Edge Calculator**: Compares calibrated model probabilities to market prices
- **Kelly Criterion**: Optimal bet sizing recommendations
- **Pattern Recognition**: Identifies stable, transitional, and uncertain weather patterns

### Telegram Bot
- **Scanner Control**: Start, stop, and monitor the edge scanner remotely
- **Forecast Lookup**: Get multi-model forecasts and accuracy rankings
- **Edge Alerts**: Receive notifications when new opportunities are found
- **Interactive UI**: Inline keyboards for quick actions

## Installation

### Requirements
- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/weather-oracle.git
cd weather-oracle
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database and collect data:
```bash
python -m src.cli collect --days 730
```

4. Train the neural network:
```bash
python -m src.cli train --epochs 50
```

## Usage

### CLI Commands

#### Weather Prediction
```bash
# Get forecast for a city
python -m src.cli predict "New York" --state NY

# Natural language weather queries
python -m src.cli ask "Will it rain in Boston tomorrow?"

# Show system status
python -m src.cli status
```

#### Model Training
```bash
# Train with default settings (50 epochs)
python -m src.cli train

# Custom training configuration
python -m src.cli train --epochs 100 --lr 0.0005 --patience 15
```

#### Data Collection
```bash
# Collect 2 years of historical data (default)
python -m src.cli collect

# Collect 1 year of data
python -m src.cli collect --days 365
```

#### Model Evaluation
```bash
# Run full evaluation on test set
python -m src.cli evaluate
```

#### Kalshi Market Integration
```bash
# Scan for weather markets
python -m src.cli scan-markets --days 7

# Find edge opportunities
python -m src.cli find-edges --min-edge 10

# Send alerts to Telegram
python -m src.cli alert-edges --min-edge 15

# Run continuous scanner
python -m src.cli watch --interval 60 --min-edge 10
```

#### Accuracy Tracking
```bash
# Show accuracy report
python -m src.cli track-accuracy

# Log forecasts from all sources
python -m src.cli track-accuracy --log

# Update with actual observations
python -m src.cli track-accuracy --update
```

#### Backtesting
```bash
# Run backtest on 30 days of historical data
python -m src.cli backtest --days 30

# Custom backtest configuration
python -m src.cli backtest --days 60 --min-edge 15 --efficiency 0.9
```

#### Tracker Daemon
```bash
# Start automated accuracy tracking
python -m src.cli start-tracker

# Generate macOS launchd plist for auto-start
python -m src.cli start-tracker --generate-plist
```

## Telegram Bot Setup

### Create a Telegram Bot

1. Message @BotFather on Telegram
2. Send `/newbot` and follow the prompts
3. Copy the bot token provided

### Get Your Chat ID

1. Message your new bot
2. Visit `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Find your chat ID in the response

### Configure Environment

Create a `.env` file in the project root:
```bash
# Telegram configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Kalshi API (optional - for authenticated endpoints)
KALSHI_CLIENT_ID=your_client_id
KALSHI_CLIENT_SECRET=your_rsa_private_key
```

### Run the Bot

```bash
python -m src.telegram.bot
```

### Available Bot Commands

**Scanner Control:**
- `/start` - Start the edge scanner
- `/stop` - Stop the edge scanner
- `/status` - Show scanner status and stats
- `/scan` - Trigger a manual scan

**Settings:**
- `/settings` - View current configuration
- `/set edge <value>` - Set minimum edge threshold
- `/set interval <minutes>` - Set scan interval
- `/set days <value>` - Set days ahead filter

**Lookups:**
- `/weather <city>` - Get 24h forecast
- `/market <ticker>` - View market details
- `/edges` - Show top edge opportunities
- `/forecast <city>` - Multi-model comparison

**Tracking:**
- `/history` - View recent alerts
- `/stats` - Show scanner statistics
- `/accuracy` - Source accuracy rankings
- `/mute` / `/unmute` - Toggle alert notifications

## Kalshi Integration Setup

Weather Oracle scans public Kalshi weather markets without authentication. For placing trades, you need API credentials.

### Public Endpoints (No Auth Required)
- Market listings and prices
- Order book data
- Market details

### Authenticated Endpoints (Requires API Key)
- Placing orders
- Account information

### Getting Kalshi API Credentials

1. Sign up at [kalshi.com](https://kalshi.com)
2. Navigate to Settings - API
3. Generate an RSA key pair
4. Add the private key to your `.env` file

## V2 Features

Weather Oracle V2 introduces significant improvements for production-grade edge detection:

### Multi-Model Ensemble

Combines 5+ weather sources with inverse MAE weighting:
- GFS (Global Forecast System)
- ICON (German weather model)
- GEM (Canadian weather model)
- NWS (National Weather Service)
- Custom LSTM neural network

### Bias Correction

Automatically tracks and corrects systematic forecast errors:
- Per-source bias tracking (positive = too hot, negative = too cold)
- Per-location calibration
- 14-day rolling window for bias estimation

### Probabilistic Calibration

Converts point forecasts to threshold probabilities:
- Gaussian CDF-based probability calculation
- Accounts for forecast uncertainty (typically 3-5F standard deviation)
- Lead time adjustment (uncertainty increases with forecast horizon)

Example: If the ensemble predicts 50F with 4F uncertainty, the probability of exceeding 53F is approximately 23%, not 0%.

### Pattern Recognition

Identifies weather patterns that affect prediction reliability:
- **Stable**: High model agreement, good confidence
- **Transitional**: Frontal passage, increased uncertainty
- **Uncertain**: Large model spread, avoid betting
- **Extreme**: Unusual temperatures, higher risk

### Kelly Criterion

Optimal bet sizing based on edge and probability:
```
Kelly fraction = (model_prob - kalshi_prob) / (1 - kalshi_prob)
```
Capped at 25% to limit risk.

## Project Structure

```
weather-oracle/
├── src/
│   ├── api/              # External API clients
│   │   ├── open_meteo.py # Weather data (current, historical, forecast)
│   │   ├── geocoding.py  # City to coordinates lookup
│   │   ├── kalshi.py     # Kalshi prediction market API
│   │   └── weather_models.py # Multi-source weather fetching
│   ├── db/
│   │   └── database.py   # SQLite operations
│   ├── data/
│   │   ├── collector.py  # Historical data collection
│   │   └── preprocessing.py # Data normalization and sequences
│   ├── model/
│   │   └── weather_net.py # LSTM encoder-decoder with attention
│   ├── training/
│   │   └── trainer.py    # Training loop with checkpointing
│   ├── inference/
│   │   ├── predictor.py  # Single-model prediction
│   │   ├── ensemble.py   # V1 ensemble (NN + API)
│   │   └── ensemble_v2.py # V2 multi-model ensemble
│   ├── evaluation/
│   │   └── metrics.py    # MAE, RMSE, accuracy metrics
│   ├── calibration/
│   │   ├── bias_correction.py # Per-source bias tracking
│   │   └── probability.py # Gaussian CDF calibration
│   ├── kalshi/
│   │   ├── scanner.py    # Market discovery and parsing
│   │   ├── edge.py       # V1 edge calculator
│   │   ├── edge_v2.py    # V2 calibrated edge calculator
│   │   └── scheduler.py  # Continuous scanning loop
│   ├── analysis/
│   │   └── patterns.py   # Weather pattern classification
│   ├── tracking/
│   │   └── forecast_tracker.py # Multi-source accuracy logging
│   ├── backtesting/
│   │   └── backtest.py   # Historical strategy testing
│   ├── daemon/
│   │   └── tracker_daemon.py # Automated accuracy tracking
│   ├── telegram/
│   │   └── bot.py        # Interactive Telegram bot
│   ├── config.py         # All configuration and paths
│   └── cli.py            # Command-line interface
├── data/                 # SQLite database and scalers
├── models/               # Saved model checkpoints
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Package metadata
```

## Model Architecture

The core neural network is an LSTM encoder-decoder with Bahdanau attention:

**Input**: 72 hours of weather data (6 features)
- Temperature
- Humidity
- Wind speed
- Precipitation
- Pressure
- Cloud cover

**Output**: 24-hour forecast (3 targets)
- Temperature
- Precipitation probability
- Wind speed

**Architecture**:
- Encoder: 2-layer LSTM (128 hidden units)
- Attention: Bahdanau-style (128 attention units)
- Decoder: 2-layer LSTM (128 hidden units)
- Total parameters: ~500,000

## Performance

Trained on 350,000+ hourly observations from 20 US cities:

| Metric | Value | Target |
|--------|-------|--------|
| Temperature MAE | 1.26F | < 2.5F |
| Temperature RMSE | 1.64F | - |
| Precipitation Accuracy | 99.0% | > 70% |
| Wind Speed MAE | 2.44 mph | - |

Improvement over persistence baseline:
- Temperature: 68% improvement
- Wind speed: 49% improvement

## Database Schema

Weather Oracle uses SQLite with the following main tables:

- `observations`: Historical hourly weather data
- `forecasts`: Model predictions with actual outcomes
- `forecast_log`: Multi-source forecast tracking
- `bias_corrections`: Per-source, per-location bias values
- `alerted_markets`: Edge alerts already sent (spam prevention)
- `scanner_state`: Bot configuration and state
- `pattern_classifications`: Weather pattern analysis

## Configuration

All configuration is in `src/config.py`:

```python
# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "weather.db"

# Training hyperparameters
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# API URLs
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for free weather API access
- [Kalshi](https://kalshi.com/) for prediction market data
- [PyTorch](https://pytorch.org/) for deep learning framework
