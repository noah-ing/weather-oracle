"""Historical Kalshi data collection for Weather Oracle.

This module collects and stores historical weather market data from Kalshi,
including market outcomes, price history, and trading activity. This data
is used to train the Oracle model to predict actual outcomes, not just
synthesize forecasts.

Key Features:
- Fetch settled markets with actual outcomes (did the event happen?)
- Get candlestick price history (how did odds move over time?)
- Store in SQLite for efficient querying
- Continuous scraping for ongoing data collection
"""

import datetime
import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.api.kalshi import KalshiClient, get_kalshi_client, Market
from src.kalshi.scanner import WeatherMarket, _parse_market
from src.config import PROJECT_ROOT


# Database path
DB_PATH = PROJECT_ROOT / "data" / "kalshi_history.db"


@dataclass
class MarketOutcome:
    """A settled market with its actual outcome."""
    ticker: str
    title: str
    event_ticker: str
    location: str
    state: Optional[str]
    target_date: Optional[str]
    condition_type: str
    threshold: Optional[float]
    comparison: Optional[str]

    # Settlement info
    settlement_time: str
    outcome: str  # "yes" or "no" - did the event happen?
    settlement_value: float  # 1.0 for yes, 0.0 for no

    # Final market state
    final_yes_price: float
    final_volume: int

    # Metadata
    collected_at: str


@dataclass
class CandlestickData:
    """Price history for a market."""
    ticker: str
    timestamp: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    period_seconds: int  # 60, 3600, 86400


@dataclass
class MarketSnapshot:
    """Point-in-time snapshot of a market."""
    ticker: str
    timestamp: str
    yes_bid: float
    yes_ask: float
    last_price: float
    volume: int
    hours_to_expiry: float

    # Weather data at snapshot time (to be filled by forecasts)
    forecast_temp: Optional[float] = None
    forecast_precip: Optional[float] = None


def init_database() -> sqlite3.Connection:
    """Initialize the SQLite database with required tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Settled markets with outcomes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_outcomes (
            ticker TEXT PRIMARY KEY,
            title TEXT,
            event_ticker TEXT,
            location TEXT,
            state TEXT,
            target_date TEXT,
            condition_type TEXT,
            threshold REAL,
            comparison TEXT,
            settlement_time TEXT,
            outcome TEXT,
            settlement_value REAL,
            final_yes_price REAL,
            final_volume INTEGER,
            collected_at TEXT
        )
    """)

    # Price history (candlesticks)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candlesticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            timestamp TEXT,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            period_seconds INTEGER,
            UNIQUE(ticker, timestamp, period_seconds)
        )
    """)

    # Market snapshots for live tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            timestamp TEXT,
            yes_bid REAL,
            yes_ask REAL,
            last_price REAL,
            volume INTEGER,
            hours_to_expiry REAL,
            forecast_temp REAL,
            forecast_precip REAL,
            UNIQUE(ticker, timestamp)
        )
    """)

    # Trade history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            trade_id TEXT UNIQUE,
            timestamp TEXT,
            price REAL,
            count INTEGER,
            taker_side TEXT
        )
    """)

    # Indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_location ON market_outcomes(location)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_condition ON market_outcomes(condition_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_date ON market_outcomes(target_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_candlesticks_ticker ON candlesticks(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON market_snapshots(ticker)")

    conn.commit()
    return conn


class KalshiHistoricalCollector:
    """Collects and stores historical Kalshi weather market data."""

    def __init__(self, client: Optional[KalshiClient] = None):
        self.client = client or get_kalshi_client()
        self.conn = init_database()

    def _request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        """Make API request with rate limiting."""
        return self.client._request(method, path, params)

    def fetch_settled_markets(self, days_back: int = 30) -> List[MarketOutcome]:
        """Fetch recently settled weather markets.

        Args:
            days_back: How many days of history to fetch

        Returns:
            List of MarketOutcome objects
        """
        outcomes = []
        seen_tickers = set()

        # Weather series to query
        weather_series = [
            # NYC
            "KXHIGHNY", "KXLOWNY", "KXLOWTNYC", "KXSNOWNY", "KXRAINNY",
            # Chicago
            "KXHIGHCHI", "KXLOWCHI", "KXLOWTCHI", "KXSNOWCHI", "KXRAINCHI",
            # Denver
            "KXHIGHDEN", "KXLOWDEN", "KXLOWTDEN", "KXSNOWDEN",
            # Miami
            "KXHIGHMIA", "KXLOWMIA", "KXLOWTMIA", "KXRAINMIA",
            # Houston
            "KXHIGHHOU", "KXLOWHOU", "KXRAINHOU",
            # LA
            "KXHIGHLAX", "KXLOWLAX", "KXLOWTLAX", "KXRAINLAX",
            # Philadelphia
            "KXHIGHPHIL", "KXLOWPHIL", "KXLOWTPHIL", "KXSNOWPHIL",
            # Austin
            "KXHIGHAUS", "KXLOWAUS", "KXLOWTAUS",
            # Boston, SF, Seattle, DC, Dallas, Detroit
            "KXSNOWBOS", "KXRAINSFO", "KXRAINSEA", "KXSNOWDC",
            # Death Valley
            "KXDVHIGH",
        ]

        print(f"  Querying {len(weather_series)} weather series...")

        for series in weather_series:
            cursor = None

            while True:
                params = {
                    "series_ticker": series,
                    "status": "settled",
                    "limit": 100,
                }
                if cursor:
                    params["cursor"] = cursor

                try:
                    response = self._request("GET", "/markets", params)
                    markets = response.get("markets", [])
                    cursor = response.get("cursor")

                    for m in markets:
                        ticker = m.get("ticker", "")

                        if ticker in seen_tickers:
                            continue
                        seen_tickers.add(ticker)

                        # Parse market info
                        market = self.client._parse_market(m)
                        parsed = _parse_market(market)

                        # Get settlement result
                        result = m.get("result", "")
                        if not result:
                            continue

                        settlement_value = 1.0 if result == "yes" else 0.0

                        outcome = MarketOutcome(
                            ticker=ticker,
                            title=market.title,
                            event_ticker=market.event_ticker,
                            location=parsed.location,
                            state=parsed.state,
                            target_date=str(parsed.target_date) if parsed.target_date else None,
                            condition_type=parsed.condition_type,
                            threshold=parsed.threshold,
                            comparison=parsed.comparison,
                            settlement_time=m.get("close_time", ""),
                            outcome=result,
                            settlement_value=settlement_value,
                            final_yes_price=market.last_price,
                            final_volume=market.volume,
                            collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        )
                        outcomes.append(outcome)

                    if not cursor or not markets:
                        break

                except Exception as e:
                    # Series might not exist
                    break

            if len(outcomes) % 100 == 0 and len(outcomes) > 0:
                print(f"    Collected {len(outcomes)} outcomes so far...")

        print(f"  Total: {len(outcomes)} settled weather markets")
        return outcomes

    def fetch_candlesticks(
        self,
        ticker: str,
        period_seconds: int = 3600,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[CandlestickData]:
        """Fetch candlestick price history for a market.

        Args:
            ticker: Market ticker
            period_seconds: Candle period (60=1min, 3600=1hr, 86400=1day)
            start_ts: Start timestamp (epoch seconds)
            end_ts: End timestamp (epoch seconds)

        Returns:
            List of CandlestickData objects
        """
        candlesticks = []

        params = {
            "ticker": ticker,
            "period_interval": period_seconds,
        }
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts

        try:
            # Use the batch endpoint for efficiency
            response = self._request("GET", "/markets/candlesticks", params)
            candles = response.get("candlesticks", [])

            for c in candles:
                candlesticks.append(CandlestickData(
                    ticker=ticker,
                    timestamp=datetime.datetime.fromtimestamp(
                        c.get("end_period_ts", 0),
                        tz=datetime.timezone.utc
                    ).isoformat(),
                    open_price=c.get("open", 0.0),
                    high_price=c.get("high", 0.0),
                    low_price=c.get("low", 0.0),
                    close_price=c.get("close", 0.0),
                    volume=c.get("volume", 0),
                    period_seconds=period_seconds,
                ))

        except Exception as e:
            print(f"Error fetching candlesticks for {ticker}: {e}")

        return candlesticks

    def fetch_trades(self, ticker: str, limit: int = 1000) -> List[Dict]:
        """Fetch trade history for a market.

        Args:
            ticker: Market ticker
            limit: Max trades to fetch

        Returns:
            List of trade dictionaries
        """
        trades = []
        cursor = None

        while len(trades) < limit:
            params = {
                "ticker": ticker,
                "limit": min(100, limit - len(trades)),
            }
            if cursor:
                params["cursor"] = cursor

            try:
                response = self._request("GET", "/markets/trades", params)
                batch = response.get("trades", [])
                trades.extend(batch)
                cursor = response.get("cursor")

                if not cursor or not batch:
                    break

            except Exception as e:
                print(f"Error fetching trades for {ticker}: {e}")
                break

        return trades

    def save_outcome(self, outcome: MarketOutcome) -> bool:
        """Save a market outcome to the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_outcomes
                (ticker, title, event_ticker, location, state, target_date,
                 condition_type, threshold, comparison, settlement_time,
                 outcome, settlement_value, final_yes_price, final_volume, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.ticker, outcome.title, outcome.event_ticker,
                outcome.location, outcome.state, outcome.target_date,
                outcome.condition_type, outcome.threshold, outcome.comparison,
                outcome.settlement_time, outcome.outcome, outcome.settlement_value,
                outcome.final_yes_price, outcome.final_volume, outcome.collected_at,
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving outcome {outcome.ticker}: {e}")
            return False

    def save_candlesticks(self, candlesticks: List[CandlestickData]) -> int:
        """Save candlesticks to database."""
        saved = 0
        cursor = self.conn.cursor()

        for c in candlesticks:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO candlesticks
                    (ticker, timestamp, open_price, high_price, low_price,
                     close_price, volume, period_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.ticker, c.timestamp, c.open_price, c.high_price,
                    c.low_price, c.close_price, c.volume, c.period_seconds,
                ))
                saved += cursor.rowcount
            except Exception as e:
                print(f"Error saving candlestick: {e}")

        self.conn.commit()
        return saved

    def save_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """Save a market snapshot to database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO market_snapshots
                (ticker, timestamp, yes_bid, yes_ask, last_price, volume,
                 hours_to_expiry, forecast_temp, forecast_precip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.ticker, snapshot.timestamp, snapshot.yes_bid,
                snapshot.yes_ask, snapshot.last_price, snapshot.volume,
                snapshot.hours_to_expiry, snapshot.forecast_temp, snapshot.forecast_precip,
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return False

    def collect_all_history(self, days_back: int = 90) -> Dict[str, int]:
        """Collect all available historical data.

        Args:
            days_back: Days of history to collect

        Returns:
            Stats dict with counts of collected items
        """
        stats = {
            "outcomes": 0,
            "candlesticks": 0,
            "trades": 0,
        }

        print(f"Collecting {days_back} days of historical data...")

        # 1. Fetch settled markets
        print("\n1. Fetching settled markets...")
        outcomes = self.fetch_settled_markets(days_back)

        for outcome in outcomes:
            if self.save_outcome(outcome):
                stats["outcomes"] += 1

        print(f"   Saved {stats['outcomes']} market outcomes")

        # 2. Fetch candlesticks for each settled market
        print("\n2. Fetching price history...")
        for i, outcome in enumerate(outcomes):
            if i % 10 == 0:
                print(f"   Processing {i+1}/{len(outcomes)}...")

            candles = self.fetch_candlesticks(outcome.ticker, period_seconds=3600)
            stats["candlesticks"] += self.save_candlesticks(candles)

            # Rate limiting
            time.sleep(0.5)

        print(f"   Saved {stats['candlesticks']} candlesticks")

        return stats

    def get_training_data(self) -> List[Dict]:
        """Get data formatted for training the Oracle model.

        Returns:
            List of training examples with features and labels
        """
        cursor = self.conn.cursor()

        # Join outcomes with candlestick history
        cursor.execute("""
            SELECT
                o.ticker,
                o.location,
                o.condition_type,
                o.threshold,
                o.comparison,
                o.target_date,
                o.outcome,
                o.settlement_value,
                o.final_yes_price,
                o.final_volume,
                c.timestamp as candle_time,
                c.close_price as market_price,
                c.volume as candle_volume
            FROM market_outcomes o
            LEFT JOIN candlesticks c ON o.ticker = c.ticker
            WHERE o.outcome IS NOT NULL
            ORDER BY o.ticker, c.timestamp
        """)

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        return [dict(zip(columns, row)) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        stats = {}

        # Count outcomes
        cursor.execute("SELECT COUNT(*) FROM market_outcomes")
        stats["total_outcomes"] = cursor.fetchone()[0]

        # Count by outcome
        cursor.execute("SELECT outcome, COUNT(*) FROM market_outcomes GROUP BY outcome")
        stats["outcomes_by_result"] = dict(cursor.fetchall())

        # Count by condition type
        cursor.execute("SELECT condition_type, COUNT(*) FROM market_outcomes GROUP BY condition_type")
        stats["outcomes_by_condition"] = dict(cursor.fetchall())

        # Count by location
        cursor.execute("SELECT location, COUNT(*) FROM market_outcomes GROUP BY location ORDER BY COUNT(*) DESC LIMIT 10")
        stats["top_locations"] = dict(cursor.fetchall())

        # Candlestick count
        cursor.execute("SELECT COUNT(*) FROM candlesticks")
        stats["total_candlesticks"] = cursor.fetchone()[0]

        # Snapshot count
        cursor.execute("SELECT COUNT(*) FROM market_snapshots")
        stats["total_snapshots"] = cursor.fetchone()[0]

        # Date range
        cursor.execute("SELECT MIN(target_date), MAX(target_date) FROM market_outcomes")
        date_range = cursor.fetchone()
        stats["date_range"] = {"min": date_range[0], "max": date_range[1]}

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


def collect_historical_data(days_back: int = 90) -> Dict[str, int]:
    """Convenience function to collect historical data.

    Args:
        days_back: Days of history to collect

    Returns:
        Collection statistics
    """
    collector = KalshiHistoricalCollector()
    try:
        stats = collector.collect_all_history(days_back)
        return stats
    finally:
        collector.close()


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]Kalshi Historical Data Collector[/bold cyan]\n")

    collector = KalshiHistoricalCollector()

    # Check existing data
    stats = collector.get_stats()
    console.print(f"[dim]Existing data: {stats['total_outcomes']} outcomes, {stats['total_candlesticks']} candlesticks[/dim]\n")

    # Collect new data
    console.print("[bold]Starting data collection (90 days)...[/bold]\n")

    try:
        results = collector.collect_all_history(days_back=90)

        console.print("\n[bold green]Collection Complete![/bold green]")

        # Show results
        table = Table(title="Collection Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        table.add_row("Market Outcomes", str(results["outcomes"]))
        table.add_row("Candlesticks", str(results["candlesticks"]))

        console.print(table)

        # Show updated stats
        stats = collector.get_stats()

        console.print("\n[bold]Database Statistics:[/bold]")
        console.print(f"  Total outcomes: {stats['total_outcomes']}")
        console.print(f"  By result: {stats['outcomes_by_result']}")
        console.print(f"  By condition: {stats['outcomes_by_condition']}")
        console.print(f"  Top locations: {list(stats['top_locations'].keys())[:5]}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        collector.close()
