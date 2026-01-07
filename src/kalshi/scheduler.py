"""Automated edge scanner with scheduler for continuous monitoring.

This module provides functionality to continuously scan Kalshi weather markets
for edge opportunities, send Telegram alerts, and log results.

Usage:
    >>> from src.kalshi.scheduler import run_scanner
    >>> run_scanner(interval_minutes=60, min_edge=10)  # Runs until Ctrl+C

Or via CLI:
    $ python -m src.cli watch --interval 60 --min-edge 10
"""

import csv
import signal
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR, DB_PATH
from src.kalshi.edge import EdgeOpportunity, find_edges
from src.telegram.bot import send_alert, send_edge_alert, send_edge_summary


# Scanner log file
SCANNER_LOG_PATH = DATA_DIR / "scanner_log.csv"

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle interrupt signal for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\nShutdown requested. Finishing current scan...")


def init_alerted_markets_table() -> None:
    """Initialize the alerted_markets table if it doesn't exist.

    Creates the table for tracking already-alerted markets to avoid spam.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerted_markets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            alerted_at TEXT NOT NULL,
            edge_pct REAL NOT NULL,
            model_prob REAL NOT NULL,
            kalshi_prob REAL NOT NULL,
            side TEXT NOT NULL,
            confidence TEXT NOT NULL,
            UNIQUE(ticker)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerted_markets_ticker
        ON alerted_markets(ticker)
    """)

    conn.commit()
    conn.close()


def is_already_alerted(ticker: str) -> bool:
    """Check if a market has already been alerted.

    Args:
        ticker: Market ticker to check

    Returns:
        True if market was already alerted, False otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM alerted_markets WHERE ticker = ?",
        (ticker,)
    )
    count = cursor.fetchone()[0]
    conn.close()

    return count > 0


def mark_as_alerted(edge: EdgeOpportunity) -> None:
    """Mark a market as alerted to avoid sending duplicate alerts.

    Args:
        edge: EdgeOpportunity that was alerted
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO alerted_markets
        (ticker, alerted_at, edge_pct, model_prob, kalshi_prob, side, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        edge.market.ticker,
        datetime.now().isoformat(),
        edge.edge_pct,
        edge.model_prob,
        edge.kalshi_prob,
        edge.side,
        edge.confidence,
    ))

    conn.commit()
    conn.close()


def clear_old_alerts(days: int = 7) -> int:
    """Clear alerts older than the specified number of days.

    Args:
        days: Number of days after which to clear old alerts

    Returns:
        Number of alerts cleared
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_str = cutoff.isoformat()

    # Delete alerts for expired markets (based on alerted_at date being old)
    # This is a simple approach - clear alerts older than N days
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    cursor.execute(
        "DELETE FROM alerted_markets WHERE alerted_at < ?",
        (cutoff_str,)
    )

    count = cursor.rowcount
    conn.commit()
    conn.close()

    return count


def get_alerted_count() -> int:
    """Get the count of alerted markets.

    Returns:
        Number of markets that have been alerted
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM alerted_markets")
    count = cursor.fetchone()[0]
    conn.close()

    return count


def init_scanner_log() -> None:
    """Initialize the scanner log CSV file if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SCANNER_LOG_PATH.exists():
        with open(SCANNER_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "markets_scanned",
                "edges_found",
                "edges_above_threshold",
                "alerts_sent",
                "new_opportunities",
                "scan_duration_sec",
                "min_edge_threshold",
            ])


def log_scan(
    markets_scanned: int,
    edges_found: int,
    edges_above_threshold: int,
    alerts_sent: int,
    new_opportunities: int,
    scan_duration: float,
    min_edge: float,
) -> None:
    """Log a scan result to the CSV file.

    Args:
        markets_scanned: Number of markets scanned
        edges_found: Total edges calculated
        edges_above_threshold: Edges above minimum threshold
        alerts_sent: Number of Telegram alerts sent
        new_opportunities: Number of new (not previously alerted) opportunities
        scan_duration: Duration of the scan in seconds
        min_edge: Minimum edge threshold used
    """
    with open(SCANNER_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            markets_scanned,
            edges_found,
            edges_above_threshold,
            alerts_sent,
            new_opportunities,
            f"{scan_duration:.2f}",
            min_edge,
        ])


def run_single_scan(
    min_edge: float = 10,
    max_series: int = 100,
    days_ahead: int = 7,
    send_alerts: bool = True,
    verbose: bool = True,
) -> dict:
    """Run a single scan for edge opportunities.

    Args:
        min_edge: Minimum edge percentage to consider
        max_series: Maximum Kalshi series to query
        days_ahead: Days ahead to include
        send_alerts: Whether to send Telegram alerts
        verbose: Whether to print progress

    Returns:
        Dict with scan statistics
    """
    start_time = time.time()

    if verbose:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting scan...")

    # Find edges
    edges = find_edges(
        min_edge=min_edge,
        max_series=max_series,
        days_ahead=days_ahead,
    )

    # Filter to new opportunities (not already alerted)
    new_edges = [e for e in edges if not is_already_alerted(e.market.ticker)]

    # Send alerts for new opportunities
    alerts_sent = 0
    if send_alerts and new_edges:
        if verbose:
            print(f"  Found {len(new_edges)} new opportunities (total: {len(edges)})")

        # Send summary if multiple new opportunities
        if len(new_edges) >= 3:
            if send_edge_summary(new_edges):
                alerts_sent += 1

        # Send individual alerts for top opportunities
        for edge in new_edges[:5]:  # Max 5 individual alerts per scan
            if send_edge_alert(edge):
                alerts_sent += 1
                mark_as_alerted(edge)
                if verbose:
                    edge_sign = "+" if edge.edge_pct > 0 else ""
                    print(f"    Alerted: {edge.market.ticker} ({edge_sign}{edge.edge_pct:.0f}% edge)")
            time.sleep(1)  # Rate limit between alerts

    elif verbose:
        if edges:
            print(f"  Found {len(edges)} opportunities, {len(new_edges)} new")
        else:
            print("  No opportunities found")

    scan_duration = time.time() - start_time

    # Log the scan
    log_scan(
        markets_scanned=max_series,  # Approximate
        edges_found=len(edges),
        edges_above_threshold=len(edges),
        alerts_sent=alerts_sent,
        new_opportunities=len(new_edges),
        scan_duration=scan_duration,
        min_edge=min_edge,
    )

    return {
        "edges_found": len(edges),
        "new_opportunities": len(new_edges),
        "alerts_sent": alerts_sent,
        "scan_duration": scan_duration,
    }


def run_scanner(
    interval_minutes: int = 60,
    min_edge: float = 10,
    max_series: int = 100,
    days_ahead: int = 7,
    clear_alerts_days: int = 7,
) -> None:
    """Run the edge scanner continuously.

    Scans weather markets at regular intervals, calculates edges,
    and sends Telegram alerts for new opportunities. Tracks already-alerted
    markets to avoid spam.

    Args:
        interval_minutes: Minutes between scans (default 60)
        min_edge: Minimum edge percentage to alert on (default 10)
        max_series: Maximum Kalshi series to query
        days_ahead: Days ahead to include in scan
        clear_alerts_days: Days after which to clear old alerts

    The scanner runs until interrupted with Ctrl+C.

    Example:
        >>> from src.kalshi.scheduler import run_scanner
        >>> run_scanner(interval_minutes=60, min_edge=10)
    """
    global _shutdown_requested
    _shutdown_requested = False

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Initialize
    init_alerted_markets_table()
    init_scanner_log()

    # Clear old alerts
    cleared = clear_old_alerts(days=clear_alerts_days)
    if cleared > 0:
        print(f"Cleared {cleared} old alerts")

    print(f"\n{'='*50}")
    print(f"Weather Oracle Edge Scanner")
    print(f"{'='*50}")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Min edge: {min_edge}%")
    print(f"Days ahead: {days_ahead}")
    print(f"Log file: {SCANNER_LOG_PATH}")
    print(f"{'='*50}")
    print("Press Ctrl+C to stop\n")

    # Send startup alert
    startup_msg = (
        f"Edge Scanner Started\n\n"
        f"Interval: {interval_minutes} min\n"
        f"Min edge: {min_edge}%\n"
        f"Days ahead: {days_ahead}"
    )
    send_alert(startup_msg)

    scan_count = 0
    total_alerts = 0
    total_opportunities = 0

    while not _shutdown_requested:
        scan_count += 1
        print(f"\n[Scan #{scan_count}]")

        try:
            result = run_single_scan(
                min_edge=min_edge,
                max_series=max_series,
                days_ahead=days_ahead,
                send_alerts=True,
                verbose=True,
            )

            total_alerts += result["alerts_sent"]
            total_opportunities += result["new_opportunities"]

            print(f"  Duration: {result['scan_duration']:.1f}s")
            print(f"  Total alerts sent: {total_alerts}")

        except KeyboardInterrupt:
            _shutdown_requested = True
            break
        except Exception as e:
            print(f"  Error during scan: {e}")

        if _shutdown_requested:
            break

        # Wait for next interval
        print(f"\nNext scan in {interval_minutes} minutes...")

        # Wait in small increments to allow for quick shutdown
        wait_seconds = interval_minutes * 60
        waited = 0
        while waited < wait_seconds and not _shutdown_requested:
            time.sleep(min(10, wait_seconds - waited))
            waited += 10

    # Shutdown
    print(f"\n{'='*50}")
    print(f"Scanner stopped")
    print(f"{'='*50}")
    print(f"Total scans: {scan_count}")
    print(f"Total alerts sent: {total_alerts}")
    print(f"Total new opportunities: {total_opportunities}")
    print(f"Log file: {SCANNER_LOG_PATH}")
    print(f"{'='*50}\n")

    # Send shutdown alert
    shutdown_msg = (
        f"Edge Scanner Stopped\n\n"
        f"Scans: {scan_count}\n"
        f"Alerts: {total_alerts}\n"
        f"Opportunities: {total_opportunities}"
    )
    send_alert(shutdown_msg)


if __name__ == "__main__":
    # Run with default settings when executed directly
    print("Starting Weather Oracle Edge Scanner...")
    print("(Use Ctrl+C to stop)\n")

    run_scanner(
        interval_minutes=1,  # 1 minute for testing
        min_edge=5,
        max_series=10,  # Fewer series for testing
        days_ahead=3,
    )
