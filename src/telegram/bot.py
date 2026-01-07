"""Telegram bot for Weather Oracle with interactive command handlers.

This module provides a persistent Telegram bot that can:
- Send alerts and edge opportunity notifications
- Accept interactive commands (/help, /start, /stop, /status, etc.)
- Run as a background process with polling
- Control the edge scanner via Telegram commands

Usage:
    >>> # As alert sender
    >>> from src.telegram.bot import send_alert, send_edge_alert
    >>> send_alert("Test message from Weather Oracle!")

    >>> # As persistent bot
    >>> python -m src.telegram.bot
"""

import asyncio
import logging
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import (
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DB_PATH, DATA_DIR
from src.kalshi.edge import EdgeOpportunity


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Rate limiting for alert sending
_last_message_time: float = 0.0
MIN_MESSAGE_INTERVAL = 1.0  # Minimum 1 second between messages


# Background scanner thread management
_scanner_thread: Optional[threading.Thread] = None
_scanner_stop_event = threading.Event()


# ============================================================================
# Database functions for bot state persistence
# ============================================================================

def _get_db_connection() -> sqlite3.Connection:
    """Get a database connection for bot state."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_scanner_state_table() -> None:
    """Create the scanner_state table if it doesn't exist."""
    conn = _get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scanner_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def get_state(key: str, default: str = "") -> str:
    """Get a state value from the scanner_state table."""
    conn = _get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM scanner_state WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()

    return row["value"] if row else default


def set_state(key: str, value: str) -> None:
    """Set a state value in the scanner_state table."""
    conn = _get_db_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT OR REPLACE INTO scanner_state (key, value, updated_at)
        VALUES (?, ?, ?)
    """, (key, value, now))

    conn.commit()
    conn.close()


def get_all_state() -> dict[str, str]:
    """Get all state values as a dictionary."""
    conn = _get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT key, value FROM scanner_state")
    rows = cursor.fetchall()
    conn.close()

    return {row["key"]: row["value"] for row in rows}


# ============================================================================
# Command handlers
# ============================================================================

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message with all available commands when /help is issued."""
    help_text = """<b>Weather Oracle Bot Commands</b>

<b>Scanner Control:</b>
/start - Start the edge scanner in background
/stop - Stop the edge scanner
/status - Show scanner status and last scan info
/scan - Trigger an immediate scan

<b>Settings:</b>
/settings - Show current configuration
/set edge &lt;value&gt; - Set minimum edge threshold
/set interval &lt;minutes&gt; - Set scan interval
/set days &lt;value&gt; - Set days ahead filter

<b>Lookups:</b>
/weather &lt;city&gt; - Get 24h forecast for a city
/market &lt;ticker&gt; - Get market details and edge
/edges - Show top edge opportunities

<b>Tracking:</b>
/history - Show last 10 alerts sent
/stats - Show scanner statistics
/mute - Temporarily disable alerts
/unmute - Re-enable alerts

<b>Other:</b>
/help - Show this help message
"""
    await update.message.reply_text(help_text, parse_mode="HTML")


# ============================================================================
# Scanner control functions
# ============================================================================

def _run_scanner_loop() -> None:
    """Run the scanner in a background thread.

    This function runs continuously until _scanner_stop_event is set.
    It uses the settings from scanner_state table.
    """
    # Lazy import to avoid circular dependency
    from src.kalshi.scheduler import run_single_scan, init_alerted_markets_table, init_scanner_log

    # Initialize tables
    init_alerted_markets_table()
    init_scanner_log()

    logger.info("Background scanner thread started")
    set_state("running", "true")
    set_state("last_started", datetime.now().isoformat())

    scan_count = 0
    total_alerts = 0

    while not _scanner_stop_event.is_set():
        try:
            # Get current settings
            interval = int(get_state("interval", "60"))
            min_edge = float(get_state("min_edge", "10"))
            max_series = int(get_state("max_series", "100"))
            days_ahead = int(get_state("days_ahead", "7"))

            # Run a scan
            scan_count += 1
            set_state("scan_count", str(scan_count))
            set_state("last_scan", datetime.now().isoformat())

            logger.info(f"Running scan #{scan_count}")

            result = run_single_scan(
                min_edge=min_edge,
                max_series=max_series,
                days_ahead=days_ahead,
                send_alerts=True,
                verbose=False,
            )

            total_alerts += result["alerts_sent"]
            set_state("total_alerts", str(total_alerts))
            set_state("last_edges_found", str(result["edges_found"]))
            set_state("last_new_opportunities", str(result["new_opportunities"]))

            logger.info(f"Scan #{scan_count} complete: {result['edges_found']} edges, {result['alerts_sent']} alerts")

        except Exception as e:
            logger.error(f"Error in scanner loop: {e}")
            set_state("last_error", str(e))

        # Wait for next interval (checking stop event every 10 seconds)
        wait_seconds = interval * 60
        waited = 0
        while waited < wait_seconds and not _scanner_stop_event.is_set():
            _scanner_stop_event.wait(timeout=10)
            waited += 10

    logger.info("Background scanner thread stopped")
    set_state("running", "false")
    set_state("last_stopped", datetime.now().isoformat())


def start_scanner() -> bool:
    """Start the background scanner thread.

    Returns:
        True if scanner was started, False if already running
    """
    global _scanner_thread, _scanner_stop_event

    if _scanner_thread is not None and _scanner_thread.is_alive():
        return False  # Already running

    # Clear stop event and start new thread
    _scanner_stop_event.clear()
    _scanner_thread = threading.Thread(target=_run_scanner_loop, daemon=True)
    _scanner_thread.start()

    return True


def stop_scanner() -> bool:
    """Stop the background scanner thread.

    Returns:
        True if scanner was stopped, False if not running
    """
    global _scanner_thread, _scanner_stop_event

    if _scanner_thread is None or not _scanner_thread.is_alive():
        set_state("running", "false")
        return False  # Not running

    # Signal stop and wait for thread to finish
    _scanner_stop_event.set()
    _scanner_thread.join(timeout=30)

    return True


def is_scanner_running() -> bool:
    """Check if the scanner is currently running.

    Returns:
        True if scanner thread is alive
    """
    global _scanner_thread
    return _scanner_thread is not None and _scanner_thread.is_alive()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command - starts the edge scanner in background."""
    if is_scanner_running():
        await update.message.reply_text(
            "🟢 Scanner is already running!\n\n"
            "Use /status to see current status or /stop to stop it.",
            parse_mode="HTML"
        )
        return

    # Get settings for confirmation message
    interval = get_state("interval", "60")
    min_edge = get_state("min_edge", "10")
    days_ahead = get_state("days_ahead", "7")

    if start_scanner():
        await update.message.reply_text(
            "🟢 <b>Scanner Started!</b>\n\n"
            f"Interval: {interval} min\n"
            f"Min edge: {min_edge}%\n"
            f"Days ahead: {days_ahead}\n\n"
            "The scanner will run in the background and send alerts "
            "when edge opportunities are found.\n\n"
            "Use /status to check progress or /stop to stop.",
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            "❌ Failed to start scanner. Please check logs.",
            parse_mode="HTML"
        )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command - stops the edge scanner."""
    if not is_scanner_running():
        await update.message.reply_text(
            "🔴 Scanner is not running.\n\n"
            "Use /start to start it.",
            parse_mode="HTML"
        )
        return

    await update.message.reply_text(
        "⏳ Stopping scanner... please wait.",
        parse_mode="HTML"
    )

    if stop_scanner():
        scan_count = get_state("scan_count", "0")
        total_alerts = get_state("total_alerts", "0")

        await update.message.reply_text(
            "🔴 <b>Scanner Stopped</b>\n\n"
            f"Total scans: {scan_count}\n"
            f"Total alerts: {total_alerts}\n\n"
            "Use /start to restart.",
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            "❌ Failed to stop scanner gracefully.",
            parse_mode="HTML"
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - shows scanner status and last scan info."""
    running = is_scanner_running()
    status_emoji = "🟢" if running else "🔴"
    status_text = "Running" if running else "Stopped"

    # Get state values
    interval = get_state("interval", "60")
    min_edge = get_state("min_edge", "10")
    days_ahead = get_state("days_ahead", "7")
    scan_count = get_state("scan_count", "0")
    total_alerts = get_state("total_alerts", "0")
    last_scan = get_state("last_scan", "")
    last_edges = get_state("last_edges_found", "0")
    last_new = get_state("last_new_opportunities", "0")
    last_error = get_state("last_error", "")

    # Calculate next scan time
    next_scan_text = ""
    if running and last_scan:
        try:
            last_scan_dt = datetime.fromisoformat(last_scan)
            from datetime import timedelta
            next_scan_dt = last_scan_dt + timedelta(minutes=int(interval))
            now = datetime.now()
            if next_scan_dt > now:
                mins_remaining = int((next_scan_dt - now).total_seconds() / 60)
                next_scan_text = f"\n⏱️ Next scan in: {mins_remaining} min"
        except Exception:
            pass

    # Format last scan time
    last_scan_text = ""
    if last_scan:
        try:
            last_scan_dt = datetime.fromisoformat(last_scan)
            last_scan_text = last_scan_dt.strftime("%H:%M:%S")
        except Exception:
            last_scan_text = last_scan

    message_lines = [
        f"{status_emoji} <b>Scanner Status: {status_text}</b>",
        "",
        "<b>Settings:</b>",
        f"  Interval: {interval} min",
        f"  Min edge: {min_edge}%",
        f"  Days ahead: {days_ahead}",
        "",
        "<b>Stats:</b>",
        f"  Total scans: {scan_count}",
        f"  Total alerts: {total_alerts}",
    ]

    if last_scan_text:
        message_lines.extend([
            "",
            "<b>Last Scan:</b>",
            f"  Time: {last_scan_text}",
            f"  Edges found: {last_edges}",
            f"  New opportunities: {last_new}",
        ])

    if next_scan_text:
        message_lines.append(next_scan_text)

    if last_error:
        message_lines.extend([
            "",
            f"⚠️ Last error: {last_error[:100]}",
        ])

    await update.message.reply_text(
        "\n".join(message_lines),
        parse_mode="HTML"
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /scan command - triggers an immediate scan."""
    await update.message.reply_text(
        "🔍 Running scan... please wait.",
        parse_mode="HTML"
    )

    try:
        # Lazy import to avoid circular dependency
        from src.kalshi.scheduler import run_single_scan

        # Get current settings
        min_edge = float(get_state("min_edge", "10"))
        max_series = int(get_state("max_series", "100"))
        days_ahead = int(get_state("days_ahead", "7"))

        result = run_single_scan(
            min_edge=min_edge,
            max_series=max_series,
            days_ahead=days_ahead,
            send_alerts=False,  # Don't send alerts for manual scan
            verbose=False,
        )

        # Update state
        set_state("last_scan", datetime.now().isoformat())
        set_state("last_edges_found", str(result["edges_found"]))
        set_state("last_new_opportunities", str(result["new_opportunities"]))

        # Report results
        edges_text = f"{result['edges_found']} edge opportunities"
        new_text = f"{result['new_opportunities']} new (not previously alerted)"
        duration_text = f"{result['scan_duration']:.1f}s"

        if result["edges_found"] > 0:
            await update.message.reply_text(
                f"✅ <b>Scan Complete</b>\n\n"
                f"📊 Found {edges_text}\n"
                f"🆕 {new_text}\n"
                f"⏱️ Duration: {duration_text}\n\n"
                f"Use /edges to see the top opportunities.",
                parse_mode="HTML"
            )
        else:
            await update.message.reply_text(
                f"✅ <b>Scan Complete</b>\n\n"
                f"No edge opportunities found above {min_edge}% threshold.\n"
                f"⏱️ Duration: {duration_text}",
                parse_mode="HTML"
            )

    except Exception as e:
        logger.error(f"Error in manual scan: {e}")
        await update.message.reply_text(
            f"❌ Scan failed: {str(e)[:100]}",
            parse_mode="HTML"
        )


# ============================================================================
# Alert sending functions (for use by scheduler)
# ============================================================================

def _get_bot() -> Optional[Bot]:
    """Get configured Telegram bot instance.

    Returns:
        Bot instance or None if not configured
    """
    if not TELEGRAM_BOT_TOKEN:
        return None
    return Bot(token=TELEGRAM_BOT_TOKEN)


def _rate_limit() -> None:
    """Apply rate limiting to avoid Telegram rate limits."""
    global _last_message_time
    elapsed = time.time() - _last_message_time
    if elapsed < MIN_MESSAGE_INTERVAL:
        time.sleep(MIN_MESSAGE_INTERVAL - elapsed)
    _last_message_time = time.time()


async def _send_message_async(bot: Bot, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
    """Send message asynchronously with retry logic.

    Args:
        bot: Telegram Bot instance
        chat_id: Chat ID to send to
        text: Message text
        parse_mode: Message parse mode ("HTML" or "Markdown")

    Returns:
        True if message sent successfully, False otherwise
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
            return True

        except RetryAfter as e:
            # Rate limited - wait the specified time
            wait_time = e.retry_after + 1
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return False

        except TimedOut:
            # Network timeout - retry after short delay
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False

        except NetworkError:
            # Network issue - retry after short delay
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return False

        except TelegramError as e:
            # Other Telegram errors - log and fail
            logger.error(f"Telegram error: {e}")
            return False

    return False


def send_alert(message: str) -> bool:
    """Send a text alert via Telegram.

    Sends a plain text message to the configured Telegram chat.
    Handles network errors and rate limits gracefully.

    Args:
        message: The message text to send

    Returns:
        True if message sent successfully, False otherwise

    Example:
        >>> from src.telegram.bot import send_alert
        >>> success = send_alert("Test from Weather Oracle!")
        >>> print(f"Message sent: {success}")
    """
    bot = _get_bot()
    if bot is None:
        print("Telegram bot not configured (TELEGRAM_BOT_TOKEN not set)")
        return False

    if not TELEGRAM_CHAT_ID:
        print("Telegram chat not configured (TELEGRAM_CHAT_ID not set)")
        return False

    _rate_limit()

    try:
        return asyncio.run(_send_message_async(bot, TELEGRAM_CHAT_ID, message, parse_mode="HTML"))
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")
        return False


def send_edge_alert(edge: EdgeOpportunity) -> bool:
    """Send a formatted edge opportunity alert via Telegram.

    Formats the edge opportunity with emojis and sends it to the
    configured Telegram chat. Includes market name, probabilities,
    edge percentage, and link to the market.

    Args:
        edge: EdgeOpportunity to send alert about

    Returns:
        True if message sent successfully, False otherwise

    Example:
        >>> from src.kalshi.edge import find_edges
        >>> from src.telegram.bot import send_edge_alert
        >>> edges = find_edges(min_edge=15)
        >>> if edges:
        ...     send_edge_alert(edges[0])
    """
    bot = _get_bot()
    if bot is None:
        print("Telegram bot not configured (TELEGRAM_BOT_TOKEN not set)")
        return False

    if not TELEGRAM_CHAT_ID:
        print("Telegram chat not configured (TELEGRAM_CHAT_ID not set)")
        return False

    # Format the message with emojis
    edge_emoji = "📈" if edge.edge_pct > 0 else "📉"
    side_emoji = "✅" if edge.side == "YES" else "❌"
    confidence_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(edge.confidence, "⚪")

    # Build market link
    market_link = f"https://kalshi.com/markets/{edge.market.ticker}"

    # Format condition description
    if edge.market.condition_type == "temp_high":
        condition = f"🌡️ High temp {edge.market.comparison} {edge.market.threshold}°F"
    elif edge.market.condition_type == "temp_low":
        condition = f"🌡️ Low temp {edge.market.comparison} {edge.market.threshold}°F"
    elif edge.market.condition_type == "rain":
        condition = "🌧️ Rain"
    elif edge.market.condition_type == "snow":
        condition = "❄️ Snow"
    else:
        condition = edge.market.condition_type

    # Build the message
    lines = [
        f"{edge_emoji} <b>Edge Opportunity Found!</b>",
        "",
        f"<b>Market:</b> {edge.market.title}",
        f"<b>Ticker:</b> <code>{edge.market.ticker}</code>",
        f"<b>Location:</b> {edge.market.location}" + (f", {edge.market.state}" if edge.market.state else ""),
        f"<b>Date:</b> {edge.market.target_date}",
        f"<b>Condition:</b> {condition}",
        "",
        f"📊 <b>Analysis:</b>",
        f"  • Kalshi Price: <b>{edge.kalshi_prob*100:.1f}%</b>",
        f"  • Model Prediction: <b>{edge.model_prob*100:.1f}%</b>",
        f"  • Edge: <b>{edge.edge_pct:+.1f}%</b>",
        f"  • Expected Value: <b>${edge.expected_value:.3f}</b> per $1",
        "",
        f"{side_emoji} <b>Recommendation:</b> {edge.side}",
        f"{confidence_emoji} <b>Confidence:</b> {edge.confidence}",
        "",
        f"🔗 <a href=\"{market_link}\">View on Kalshi</a>",
    ]

    message = "\n".join(lines)

    _rate_limit()

    try:
        return asyncio.run(_send_message_async(bot, TELEGRAM_CHAT_ID, message, parse_mode="HTML"))
    except Exception as e:
        print(f"Failed to send Telegram edge alert: {e}")
        return False


def format_edge_summary(edges: list[EdgeOpportunity]) -> str:
    """Format a summary of multiple edge opportunities.

    Args:
        edges: List of EdgeOpportunity objects

    Returns:
        Formatted HTML string for Telegram
    """
    if not edges:
        return "No edge opportunities found above threshold."

    lines = [
        f"🎯 <b>Found {len(edges)} Edge Opportunities</b>",
        "",
    ]

    for i, edge in enumerate(edges[:10], 1):  # Max 10 in summary
        edge_emoji = "📈" if edge.edge_pct > 0 else "📉"
        lines.append(
            f"{i}. {edge_emoji} <b>{edge.market.ticker[:15]}</b> | "
            f"{edge.edge_pct:+.0f}% edge | {edge.side}"
        )

    if len(edges) > 10:
        lines.append(f"\n... and {len(edges) - 10} more")

    return "\n".join(lines)


def send_edge_summary(edges: list[EdgeOpportunity]) -> bool:
    """Send a summary of edge opportunities via Telegram.

    Args:
        edges: List of EdgeOpportunity objects

    Returns:
        True if message sent successfully, False otherwise
    """
    message = format_edge_summary(edges)
    return send_alert(message)


# ============================================================================
# Bot application
# ============================================================================

def create_application() -> Application:
    """Create and configure the Telegram bot application.

    Returns:
        Configured Application instance
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not configured in .env")

    # Initialize database state table
    init_scanner_state_table()

    # Set default state values if not set
    if not get_state("interval"):
        set_state("interval", "60")
    if not get_state("min_edge"):
        set_state("min_edge", "10")
    if not get_state("max_series"):
        set_state("max_series", "100")
    if not get_state("days_ahead"):
        set_state("days_ahead", "7")

    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("help", help_command))

    # Scanner control commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("scan", scan_command))

    return application


def run_bot() -> None:
    """Run the Telegram bot with polling.

    This function blocks and runs the bot until stopped.
    """
    logger.info("Starting Weather Oracle Telegram bot...")

    try:
        application = create_application()

        # Set initial state
        set_state("running", "false")
        set_state("last_started", datetime.now().isoformat())

        logger.info("Bot is now polling for updates. Press Ctrl+C to stop.")

        # Run the bot
        application.run_polling(allowed_updates=Update.ALL_TYPES)

    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    # When run directly, start the persistent bot
    print("=" * 60)
    print("Weather Oracle Telegram Bot")
    print("=" * 60)

    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        print("Please configure your bot token in the .env file.")
        exit(1)

    if not TELEGRAM_CHAT_ID:
        print("WARNING: TELEGRAM_CHAT_ID not set in .env")
        print("Alerts may not work, but bot commands will still function.")

    print(f"Bot token configured: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"Chat ID configured: {TELEGRAM_CHAT_ID or '(not set)'}")
    print()
    print("Starting bot... Send /help to see available commands.")
    print()

    run_bot()
