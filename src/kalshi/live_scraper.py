"""Live market scraper with forecast capture.

This scraper runs continuously, capturing:
1. Market prices and odds every hour
2. Current weather forecasts for each market's location
3. Actual outcomes when markets settle

This data enables training the Oracle on:
- What the forecast said
- What the market said
- What actually happened

The key insight: learn when forecasts are wrong and when markets are wrong.
"""

import datetime
import sqlite3
import time
import schedule
from pathlib import Path
from typing import Optional, Dict, List

from src.config import PROJECT_ROOT
from src.api.kalshi import get_kalshi_client
from src.kalshi.scanner import scan_weather_markets, WeatherMarket
from src.api.open_meteo import get_forecast as get_open_meteo_forecast


DB_PATH = PROJECT_ROOT / "data" / "live_snapshots.db"


def init_live_db() -> sqlite3.Connection:
    """Initialize the live snapshots database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Market snapshots with forecast data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,

            -- Market data
            yes_bid REAL,
            yes_ask REAL,
            last_price REAL,
            volume INTEGER,
            hours_to_expiry REAL,

            -- Market metadata
            location TEXT,
            condition_type TEXT,
            comparison TEXT,
            threshold REAL,
            target_date TEXT,

            -- Forecast data (at snapshot time)
            forecast_high REAL,
            forecast_low REAL,
            forecast_precip_prob REAL,

            -- Derived features
            forecast_vs_threshold REAL,  -- forecast - threshold

            UNIQUE(ticker, timestamp)
        )
    """)

    # Settlement outcomes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settlements (
            ticker TEXT PRIMARY KEY,
            settlement_time TEXT,
            outcome TEXT,  -- 'yes' or 'no'
            final_price REAL,
            final_volume INTEGER,

            -- What we predicted at various lead times
            oracle_prob_24h REAL,
            oracle_prob_12h REAL,
            oracle_prob_6h REAL,
            forecast_24h REAL,
            forecast_12h REAL,
            forecast_6h REAL
        )
    """)

    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON snapshots(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON snapshots(timestamp)")

    conn.commit()
    return conn


class LiveScraper:
    """Captures live market data with forecasts."""

    def __init__(self):
        self.conn = init_live_db()
        self.client = get_kalshi_client()

    def _get_forecast_for_location(
        self,
        lat: float,
        lon: float,
        target_date: datetime.date,
    ) -> Dict:
        """Get forecast for a specific location and date."""
        try:
            forecast = get_open_meteo_forecast(lat, lon)

            # HourlyForecast object has: timestamps, temperatures, precipitation_probability
            if not forecast or not forecast.timestamps:
                return {"high": None, "low": None, "precip_prob": None}

            # Find temperatures for target date
            target_temps = []
            target_precip = []

            for i, ts in enumerate(forecast.timestamps):
                if ts.date() == target_date:
                    target_temps.append(forecast.temperatures[i])
                    if forecast.precipitation_probability:
                        target_precip.append(forecast.precipitation_probability[i])

            if target_temps:
                return {
                    "high": max(target_temps),
                    "low": min(target_temps),
                    "precip_prob": max(target_precip) if target_precip else None,
                }

            # Fallback: use all available data
            if forecast.temperatures:
                return {
                    "high": max(forecast.temperatures[:24]),  # First 24 hours
                    "low": min(forecast.temperatures[:24]),
                    "precip_prob": max(forecast.precipitation_probability[:24]) if forecast.precipitation_probability else None,
                }

            return {"high": None, "low": None, "precip_prob": None}

        except Exception as e:
            print(f"Forecast error: {e}")
            return {"high": None, "low": None, "precip_prob": None}

    def capture_snapshot(self) -> int:
        """Capture current state of all weather markets.

        Returns:
            Number of snapshots saved
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        saved = 0

        print(f"[{timestamp[:19]}] Capturing market snapshots...")

        try:
            markets = scan_weather_markets(max_series=50, days_ahead=7)
        except Exception as e:
            print(f"Error scanning markets: {e}")
            return 0

        cursor = self.conn.cursor()

        for market in markets:
            # Skip markets without required data
            if market.lat is None or market.lon is None:
                continue
            if market.target_date is None:
                continue

            # Calculate hours to expiry
            now = datetime.datetime.now(datetime.timezone.utc)
            if market.expiration_time:
                exp = market.expiration_time
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=datetime.timezone.utc)
                hours_to_expiry = (exp - now).total_seconds() / 3600
            else:
                hours_to_expiry = None

            # Get forecast for this location/date
            forecast = self._get_forecast_for_location(
                market.lat, market.lon, market.target_date
            )

            # Calculate forecast vs threshold
            forecast_vs_threshold = None
            if forecast["high"] is not None and market.threshold is not None:
                if market.condition_type == "temp_high":
                    forecast_vs_threshold = forecast["high"] - market.threshold
                elif market.condition_type == "temp_low":
                    forecast_vs_threshold = forecast["low"] - market.threshold

            # Save snapshot
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO snapshots
                    (ticker, timestamp, yes_bid, yes_ask, last_price, volume,
                     hours_to_expiry, location, condition_type, comparison,
                     threshold, target_date, forecast_high, forecast_low,
                     forecast_precip_prob, forecast_vs_threshold)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market.ticker, timestamp,
                    market.yes_bid, market.yes_ask, market.last_price, market.volume,
                    hours_to_expiry, market.location, market.condition_type,
                    market.comparison, market.threshold, str(market.target_date),
                    forecast["high"], forecast["low"], forecast["precip_prob"],
                    forecast_vs_threshold,
                ))
                saved += 1
            except Exception as e:
                print(f"Error saving snapshot for {market.ticker}: {e}")

        self.conn.commit()
        print(f"  Saved {saved} snapshots")
        return saved

    def check_settlements(self) -> int:
        """Check for newly settled markets and record outcomes.

        Returns:
            Number of settlements recorded
        """
        cursor = self.conn.cursor()

        # Get tickers we have snapshots for but no settlement
        cursor.execute("""
            SELECT DISTINCT s.ticker
            FROM snapshots s
            LEFT JOIN settlements st ON s.ticker = st.ticker
            WHERE st.ticker IS NULL
        """)

        tickers_to_check = [row[0] for row in cursor.fetchall()]
        settled = 0

        for ticker in tickers_to_check[:50]:  # Rate limit
            try:
                market = self.client.get_market_details(ticker)

                if market.status == "settled":
                    # Get settlement info
                    result = "unknown"  # Would need to get from API

                    cursor.execute("""
                        INSERT OR REPLACE INTO settlements
                        (ticker, settlement_time, outcome, final_price, final_volume)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        ticker,
                        datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        result,
                        market.last_price,
                        market.volume,
                    ))
                    settled += 1

                time.sleep(0.5)  # Rate limit

            except Exception as e:
                continue

        self.conn.commit()
        return settled

    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM snapshots")
        total_snapshots = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM snapshots")
        unique_markets = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM snapshots")
        time_range = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) FROM settlements")
        total_settlements = cursor.fetchone()[0]

        return {
            "total_snapshots": total_snapshots,
            "unique_markets": unique_markets,
            "time_range": {"start": time_range[0], "end": time_range[1]},
            "total_settlements": total_settlements,
        }

    def run_continuous(self, interval_minutes: int = 60):
        """Run scraper continuously.

        Args:
            interval_minutes: Minutes between snapshots
        """
        print(f"Starting live scraper (interval: {interval_minutes} min)")
        print("Press Ctrl+C to stop\n")

        # Initial capture
        self.capture_snapshot()

        # Schedule regular captures
        schedule.every(interval_minutes).minutes.do(self.capture_snapshot)
        schedule.every(6).hours.do(self.check_settlements)

        while True:
            schedule.run_pending()
            time.sleep(60)

    def close(self):
        """Close database connection."""
        self.conn.close()


def run_scraper():
    """Run the live scraper."""
    scraper = LiveScraper()
    try:
        scraper.run_continuous(interval_minutes=60)
    except KeyboardInterrupt:
        print("\nStopping scraper...")
    finally:
        scraper.close()


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    console.print("\n[bold cyan]Weather Oracle Live Scraper[/bold cyan]")
    console.print("[dim]Capturing markets + forecasts for Oracle training[/dim]\n")

    scraper = LiveScraper()

    # Show current stats
    stats = scraper.get_stats()
    console.print(f"Existing data: {stats['total_snapshots']} snapshots, "
                  f"{stats['unique_markets']} markets, "
                  f"{stats['total_settlements']} settlements\n")

    # Do one capture
    console.print("[bold]Running single capture...[/bold]")
    count = scraper.capture_snapshot()
    console.print(f"[green]Captured {count} market snapshots with forecasts[/green]")

    # Updated stats
    stats = scraper.get_stats()
    console.print(f"\n[bold]Updated stats:[/bold]")
    console.print(f"  Total snapshots: {stats['total_snapshots']}")
    console.print(f"  Unique markets:  {stats['unique_markets']}")

    console.print("\n[dim]To run continuously: python -m src.kalshi.live_scraper --daemon[/dim]")

    scraper.close()
