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

from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
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
/forecast &lt;city&gt; - Multi-model forecast with regional NN (V3)
/market &lt;ticker&gt; - Get market details and edge
/edges - Show calibrated edge opportunities (V2)

<b>V2 Analytics:</b>
/accuracy - Show source accuracy rankings
/confidence &lt;city&gt; - Model agreement for location

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
            muted = get_state("muted", "false") == "true"

            # Run a scan
            scan_count += 1
            set_state("scan_count", str(scan_count))
            set_state("last_scan", datetime.now().isoformat())

            logger.info(f"Running scan #{scan_count}" + (" (muted)" if muted else ""))

            result = run_single_scan(
                min_edge=min_edge,
                max_series=max_series,
                days_ahead=days_ahead,
                send_alerts=not muted,  # Don't send alerts when muted
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

    # Get mute state for button display
    muted = get_state("muted", "false") == "true"

    # Build inline keyboard based on current state
    buttons = []

    # First row: Start or Stop button
    if running:
        buttons.append([InlineKeyboardButton("🛑 Stop", callback_data="stop_scanner")])
    else:
        buttons.append([InlineKeyboardButton("▶️ Start", callback_data="start_scanner")])

    # Second row: Scan Now and Mute/Unmute
    buttons.append([
        InlineKeyboardButton("🔍 Scan Now", callback_data="scan_now"),
        InlineKeyboardButton("🔇 Mute" if not muted else "🔊 Unmute",
                           callback_data="mute_alerts" if not muted else "unmute_alerts")
    ])

    reply_markup = InlineKeyboardMarkup(buttons)

    await update.message.reply_text(
        "\n".join(message_lines),
        parse_mode="HTML",
        reply_markup=reply_markup
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settings command - shows current configuration."""
    # Get all current settings
    interval = get_state("interval", "60")
    min_edge = get_state("min_edge", "10")
    days_ahead = get_state("days_ahead", "7")
    max_series = get_state("max_series", "100")

    message = """⚙️ <b>Current Settings</b>

<b>Edge Threshold:</b> {min_edge}%
<i>Minimum edge percentage to trigger alerts</i>

<b>Scan Interval:</b> {interval} minutes
<i>Time between automatic scans</i>

<b>Days Ahead:</b> {days_ahead} days
<i>Only show markets expiring within this period</i>

<b>Max Series:</b> {max_series}
<i>Maximum market series to scan</i>

<b>To change settings:</b>
/set edge &lt;value&gt; - Set edge threshold (1-100)
/set interval &lt;minutes&gt; - Set scan interval (1-1440)
/set days &lt;value&gt; - Set days ahead filter (1-30)
/set max_series &lt;value&gt; - Set max series (1-200)
""".format(min_edge=min_edge, interval=interval, days_ahead=days_ahead, max_series=max_series)

    await update.message.reply_text(message, parse_mode="HTML")


async def set_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /set command - changes settings.

    Usage:
        /set edge <value> - Set minimum edge threshold (1-100)
        /set interval <minutes> - Set scan interval (1-1440)
        /set days <value> - Set days ahead filter (1-30)
        /set max_series <value> - Set max series (1-200)
    """
    # Parse the command arguments
    args = context.args

    if not args or len(args) < 2:
        await update.message.reply_text(
            "❌ <b>Invalid usage</b>\n\n"
            "Usage:\n"
            "/set edge &lt;value&gt; - Set edge threshold (1-100)\n"
            "/set interval &lt;minutes&gt; - Set scan interval (1-1440)\n"
            "/set days &lt;value&gt; - Set days ahead (1-30)\n"
            "/set max_series &lt;value&gt; - Set max series (1-200)\n\n"
            "Example: <code>/set edge 15</code>",
            parse_mode="HTML"
        )
        return

    setting_name = args[0].lower()
    value_str = args[1]

    # Validate and parse value
    try:
        value = float(value_str) if setting_name == "edge" else int(value_str)
    except ValueError:
        await update.message.reply_text(
            f"❌ Invalid value: <code>{value_str}</code>\n"
            "Please enter a valid number.",
            parse_mode="HTML"
        )
        return

    # Define valid ranges for each setting
    setting_configs = {
        "edge": {
            "key": "min_edge",
            "min": 1,
            "max": 100,
            "unit": "%",
            "display_name": "Edge Threshold",
        },
        "interval": {
            "key": "interval",
            "min": 1,
            "max": 1440,  # Max 24 hours
            "unit": " minutes",
            "display_name": "Scan Interval",
        },
        "days": {
            "key": "days_ahead",
            "min": 1,
            "max": 30,
            "unit": " days",
            "display_name": "Days Ahead",
        },
        "max_series": {
            "key": "max_series",
            "min": 1,
            "max": 200,
            "unit": "",
            "display_name": "Max Series",
        },
    }

    if setting_name not in setting_configs:
        valid_settings = ", ".join(setting_configs.keys())
        await update.message.reply_text(
            f"❌ Unknown setting: <code>{setting_name}</code>\n\n"
            f"Valid settings: {valid_settings}",
            parse_mode="HTML"
        )
        return

    config = setting_configs[setting_name]

    # Validate range
    if value < config["min"] or value > config["max"]:
        await update.message.reply_text(
            f"❌ Invalid value for {config['display_name']}\n\n"
            f"Value must be between {config['min']} and {config['max']}\n"
            f"You entered: {value}",
            parse_mode="HTML"
        )
        return

    # Get old value for confirmation
    old_value = get_state(config["key"], "")

    # Set the new value
    set_state(config["key"], str(value))

    # Confirm the change
    await update.message.reply_text(
        f"✅ <b>{config['display_name']} Updated</b>\n\n"
        f"Old value: {old_value}{config['unit']}\n"
        f"New value: {value}{config['unit']}\n\n"
        "Use /settings to see all current settings.",
        parse_mode="HTML"
    )


async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /weather command - shows 24h forecast for a city.

    Usage: /weather <city> or /weather NYC
    """
    args = context.args

    if not args:
        await update.message.reply_text(
            "❌ <b>Usage:</b> /weather &lt;city&gt;\n\n"
            "Examples:\n"
            "<code>/weather NYC</code>\n"
            "<code>/weather Chicago</code>\n"
            "<code>/weather Los Angeles</code>",
            parse_mode="HTML"
        )
        return

    # Join args to handle multi-word city names like "New York"
    city_input = " ".join(args).strip()

    await update.message.reply_text(
        f"🔍 Fetching forecast for <b>{city_input}</b>...",
        parse_mode="HTML"
    )

    try:
        # Lazy import to avoid circular dependency
        from src.inference.predictor import WeatherPredictor

        predictor = WeatherPredictor()

        # Handle common abbreviations
        city_map = {
            "NYC": ("New York", "NY"),
            "LA": ("Los Angeles", "CA"),
            "SF": ("San Francisco", "CA"),
            "CHI": ("Chicago", "IL"),
            "DC": ("Washington", "DC"),
            "PHILLY": ("Philadelphia", "PA"),
        }

        city_upper = city_input.upper()
        if city_upper in city_map:
            city, state = city_map[city_upper]
        else:
            city = city_input
            state = None

        forecast = predictor.predict_city(city, state)

        if forecast is None:
            await update.message.reply_text(
                f"❌ Could not find city: <b>{city_input}</b>\n\n"
                "Try using full city name or common abbreviation (NYC, LA, SF, CHI).",
                parse_mode="HTML"
            )
            return

        # Calculate high/low for the day
        temps = [h.temperature for h in forecast.hourly]
        temp_high = max(temps)
        temp_low = min(temps)

        # Max rain chance
        precips = [h.precip_probability for h in forecast.hourly]
        max_rain = max(precips)

        # Max wind
        winds = [h.wind_speed for h in forecast.hourly]
        max_wind = max(winds)

        # Find hours with high confidence (smallest confidence interval)
        avg_confidence = sum(
            h.temperature_high - h.temperature_low for h in forecast.hourly
        ) / len(forecast.hourly)

        # Build message
        lines = [
            f"🌤️ <b>24h Forecast for {forecast.location}</b>",
            "",
            f"📅 Generated: {forecast.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"🌡️ <b>Temperature:</b>",
            f"   High: <b>{temp_high:.0f}°F</b>",
            f"   Low: <b>{temp_low:.0f}°F</b>",
            "",
            f"🌧️ <b>Rain Chance:</b> {max_rain:.0f}%",
            f"💨 <b>Max Wind:</b> {max_wind:.0f} mph",
            f"📊 <b>Confidence:</b> ±{avg_confidence/2:.1f}°F",
            "",
            "<b>Hourly Preview:</b>",
        ]

        # Show 6 hours with compact format
        for h in forecast.hourly[:6]:
            time_str = h.timestamp.strftime("%H:%M")
            rain_icon = "🌧️" if h.precip_probability > 50 else "☀️"
            lines.append(
                f"{time_str} | {h.temperature:.0f}°F {rain_icon} {h.precip_probability:.0f}%"
            )

        if len(forecast.hourly) > 6:
            lines.append(f"... +{len(forecast.hourly) - 6} more hours")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except FileNotFoundError:
        await update.message.reply_text(
            "❌ Model not trained yet.\n\n"
            "Run <code>python -m src.training.trainer</code> first.",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error in /weather command: {e}")
        await update.message.reply_text(
            f"❌ Error fetching forecast: {str(e)[:100]}",
            parse_mode="HTML"
        )


async def market_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /market command - shows market details and model edge.

    Usage: /market <ticker>
    """
    args = context.args

    if not args:
        await update.message.reply_text(
            "❌ <b>Usage:</b> /market &lt;ticker&gt;\n\n"
            "Example:\n"
            "<code>/market KXHIGHNY-26JAN09-T35</code>",
            parse_mode="HTML"
        )
        return

    ticker = args[0].upper()

    await update.message.reply_text(
        f"🔍 Fetching market <b>{ticker}</b>...",
        parse_mode="HTML"
    )

    try:
        # Lazy imports
        from src.api.kalshi import KalshiClient
        from src.kalshi.scanner import scan_weather_markets
        from src.kalshi.edge import calculate_edge
        from src.inference.ensemble import EnsemblePredictor

        client = KalshiClient()

        # Get market details
        try:
            market_data = client.get_market_details(ticker)
        except Exception as e:
            await update.message.reply_text(
                f"❌ Market not found: <b>{ticker}</b>\n\n"
                f"Error: {str(e)[:100]}",
                parse_mode="HTML"
            )
            return

        # Get orderbook
        try:
            orderbook = client.get_orderbook(ticker)
            bid_levels = len(orderbook.yes_bids)
            ask_levels = len(orderbook.no_bids)
        except Exception:
            bid_levels = 0
            ask_levels = 0

        # Try to find parsed weather market and calculate edge
        edge_info = ""
        try:
            # Scan markets to find this specific one
            predictor = EnsemblePredictor()
            markets = scan_weather_markets(max_series=100, days_ahead=30)

            for wm in markets:
                if wm.ticker == ticker:
                    edge = calculate_edge(wm, predictor)
                    if edge:
                        edge_emoji = "📈" if edge.edge_pct > 0 else "📉"
                        side_emoji = "✅" if edge.side == "YES" else "❌"
                        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(edge.confidence, "⚪")

                        edge_info = f"""
<b>Model Analysis:</b>
   Kalshi Prob: <b>{edge.kalshi_prob*100:.1f}%</b>
   Model Prob: <b>{edge.model_prob*100:.1f}%</b>
   {edge_emoji} Edge: <b>{edge.edge_pct:+.1f}%</b>
   EV: <b>${edge.expected_value:.3f}</b>/contract
   {side_emoji} Side: <b>{edge.side}</b>
   {conf_emoji} Confidence: {edge.confidence}
"""
                    break
        except Exception as e:
            logger.warning(f"Could not calculate edge for {ticker}: {e}")

        # Format expiration
        exp_str = "N/A"
        if market_data.expiration_time:
            exp_str = market_data.expiration_time.strftime("%Y-%m-%d %H:%M")

        # Format close time
        close_str = "N/A"
        if market_data.close_time:
            close_str = market_data.close_time.strftime("%Y-%m-%d %H:%M")

        # Build message
        lines = [
            f"📊 <b>Market: {ticker}</b>",
            "",
            f"<b>Title:</b> {market_data.title}",
            f"<b>Status:</b> {market_data.status}",
            "",
            "<b>Prices:</b>",
            f"   Yes: ${market_data.yes_bid:.2f} / ${market_data.yes_ask:.2f}",
            f"   No: ${market_data.no_bid:.2f} / ${market_data.no_ask:.2f}",
            f"   Last: ${market_data.last_price:.2f}",
            "",
            "<b>Volume:</b>",
            f"   Total: {market_data.volume:,}",
            f"   24h: {market_data.volume_24h:,}",
            f"   Open Interest: {market_data.open_interest:,}",
            "",
            "<b>Timing:</b>",
            f"   Closes: {close_str}",
            f"   Expires: {exp_str}",
        ]

        if edge_info:
            lines.append(edge_info)
        else:
            lines.append("\n<i>Edge calculation not available for this market.</i>")

        lines.append(f"\n🔗 <a href=\"https://kalshi.com/markets/{ticker}\">View on Kalshi</a>")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in /market command: {e}")
        await update.message.reply_text(
            f"❌ Error fetching market: {str(e)[:100]}",
            parse_mode="HTML"
        )


async def edges_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /edges command - shows top 5 calibrated edge opportunities (V2)."""
    await update.message.reply_text(
        "🔍 Finding calibrated edge opportunities...",
        parse_mode="HTML"
    )

    try:
        # Use calibrated edges (V2)
        from src.kalshi.edge_v2 import find_calibrated_edges

        # Get current settings
        min_edge = float(get_state("min_edge", "10"))
        max_series = int(get_state("max_series", "100"))
        days_ahead = int(get_state("days_ahead", "7"))

        edges = find_calibrated_edges(
            min_edge=min_edge,
            max_series=max_series,
            days_ahead=days_ahead,
        )

        if not edges:
            await update.message.reply_text(
                f"No calibrated edge opportunities found above {min_edge}% threshold.\n\n"
                "Try lowering the threshold with:\n"
                "<code>/set edge 5</code>",
                parse_mode="HTML"
            )
            return

        # Show top 5
        top_edges = edges[:5]

        lines = [
            f"🎯 <b>Top {len(top_edges)} Calibrated Edge Opportunities (V2)</b>",
            f"<i>(min edge: {min_edge}%, {days_ahead} days ahead)</i>",
            "",
        ]

        for i, edge in enumerate(top_edges, 1):
            edge_emoji = "📈" if edge.edge_pct > 0 else "📉"
            side_emoji = "✅" if edge.side == "YES" else "❌"
            conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴", "VERY_LOW": "⚫"}.get(edge.confidence_level, "⚪")

            # Condition type icon
            cond_emoji = {
                "temp_high": "🌡️",
                "temp_low": "🌡️",
                "rain": "🌧️",
                "snow": "❄️",
                "hurricane": "🌀",
            }.get(edge.market.condition_type, "🌤️")

            date_str = str(edge.market.target_date) if edge.market.target_date else "N/A"

            lines.append(
                f"<b>{i}. {edge.market.ticker[:20]}</b>"
            )
            lines.append(
                f"   {cond_emoji} {edge.market.location} | {date_str}"
            )
            lines.append(
                f"   {edge_emoji} Edge: <b>{edge.edge_pct:+.1f}%</b> | "
                f"EV: ${edge.expected_value:.2f}"
            )
            lines.append(
                f"   {side_emoji} {edge.side} | Kelly: {edge.kelly_fraction:.1%} | {conf_emoji} {edge.confidence_level}"
            )
            if edge.sources_used:
                lines.append(f"   📊 Sources: {', '.join(edge.sources_used[:3])}")
            lines.append("")

        if len(edges) > 5:
            lines.append(f"<i>... and {len(edges) - 5} more opportunities</i>")

        lines.append("\nUse <code>/market TICKER</code> for details.")

        # Add inline keyboard with Scan Now and Settings buttons
        keyboard = [
            [InlineKeyboardButton("🔄 Scan Now", callback_data="scan_now")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="show_settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text("\n".join(lines), parse_mode="HTML", reply_markup=reply_markup)

    except Exception as e:
        logger.error(f"Error in /edges command: {e}")
        await update.message.reply_text(
            f"❌ Error finding edges: {str(e)[:100]}",
            parse_mode="HTML"
        )


# ============================================================================
# V2 Command handlers - Multi-model and calibration features
# ============================================================================

async def forecast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /forecast command - shows multi-model comparison forecast (V2).

    Usage: /forecast <city>
    """
    args = context.args

    if not args:
        await update.message.reply_text(
            "❌ <b>Usage:</b> /forecast &lt;city&gt;\n\n"
            "Example:\n"
            "<code>/forecast NYC</code>\n"
            "<code>/forecast Chicago</code>",
            parse_mode="HTML"
        )
        return

    city_input = " ".join(args).strip()

    await update.message.reply_text(
        f"🔍 Fetching multi-model forecast for <b>{city_input}</b>...",
        parse_mode="HTML"
    )

    try:
        from src.inference.ensemble_v2 import predict_ensemble

        # Handle common abbreviations
        city_map = {
            "NYC": "New York, NY",
            "LA": "Los Angeles, CA",
            "SF": "San Francisco, CA",
            "CHI": "Chicago, IL",
            "DC": "Washington, DC",
            "PHILLY": "Philadelphia, PA",
            "MIAMI": "Miami, FL",
            "DENVER": "Denver, CO",
        }

        city_upper = city_input.upper()
        location = city_map.get(city_upper, city_input)

        ensemble = predict_ensemble(location, include_nn=True)

        if ensemble is None:
            await update.message.reply_text(
                f"❌ Could not get forecast for: <b>{city_input}</b>\n\n"
                "Try using full city name (e.g., 'New York, NY').",
                parse_mode="HTML"
            )
            return

        # Check if regional_nn is among sources
        has_regional = any(c.source == "regional_nn" for c in ensemble.contributions)
        regional_info = ""
        if has_regional:
            regional_contrib = next(c for c in ensemble.contributions if c.source == "regional_nn")
            regional_info = f"\n🧠 <b>Regional NN (V3):</b> {regional_contrib.corrected_high:.1f}°F / {regional_contrib.corrected_low:.1f}°F (wt: {regional_contrib.weight:.0%})"

        # Build multi-model comparison message
        lines = [
            f"🌡️ <b>Multi-Model Forecast: {location}</b>",
            "",
            f"<b>High Temperature:</b> {ensemble.high_temp:.1f}°F (±{ensemble.high_std:.1f}°F)",
            f"<b>Low Temperature:</b> {ensemble.low_temp:.1f}°F (±{ensemble.low_std:.1f}°F)",
            f"<b>Precipitation:</b> {ensemble.precip_probability:.0f}%",
            "",
            f"📊 <b>Confidence:</b> {ensemble.confidence:.0%}",
            f"📈 <b>Sources Used:</b> {ensemble.total_sources}",
        ]

        # Highlight regional NN if available
        if regional_info:
            lines.append(regional_info)

        lines.extend([
            "",
            "<b>Source Breakdown:</b>",
        ])

        # Show per-source contributions
        for contrib in ensemble.contributions:
            # Add special indicator for regional_nn
            source_name = contrib.source
            if contrib.source == "regional_nn":
                source_name = "regional_nn (V3)"
            lines.append(
                f"  - {source_name}: {contrib.raw_high:.1f}°F -> "
                f"{contrib.corrected_high:.1f}°F (wt: {contrib.weight:.0%})"
            )

        # Add model agreement assessment
        if ensemble.confidence >= 0.8:
            agreement = "🟢 High agreement - models align well"
        elif ensemble.confidence >= 0.5:
            agreement = "🟡 Moderate agreement - some uncertainty"
        else:
            agreement = "🔴 Low agreement - significant uncertainty"

        lines.extend([
            "",
            f"<b>Model Agreement:</b> {agreement}",
        ])

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in /forecast command: {e}")
        await update.message.reply_text(
            f"❌ Error fetching forecast: {str(e)[:100]}",
            parse_mode="HTML"
        )


async def accuracy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /accuracy command - shows recent accuracy by source (V2).

    Displays MAE rankings for each forecast source over the last 7 days.
    """
    await update.message.reply_text(
        "📊 Fetching accuracy rankings...",
        parse_mode="HTML"
    )

    try:
        from src.tracking.forecast_tracker import get_accuracy_by_source

        accuracy = get_accuracy_by_source(days=7)

        if not accuracy:
            await update.message.reply_text(
                "📊 <b>No Accuracy Data Yet</b>\n\n"
                "Accuracy data is collected when forecasts are verified against actuals.\n\n"
                "Run <code>python -m src.cli track-accuracy --log --update</code> to log forecasts.",
                parse_mode="HTML"
            )
            return

        lines = [
            "📊 <b>Source Accuracy Rankings (7-day MAE)</b>",
            "",
            "Lower MAE = Better accuracy",
            "",
        ]

        # Sort by MAE (lower is better)
        sorted_accuracy = sorted(accuracy.items(), key=lambda x: x[1])

        for i, (source, mae) in enumerate(sorted_accuracy, 1):
            # Medal emoji for top 3
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")

            # Color code based on MAE
            if mae <= 2.0:
                quality = "🟢"  # Excellent
            elif mae <= 4.0:
                quality = "🟡"  # Good
            else:
                quality = "🔴"  # Needs improvement

            lines.append(f"{medal} <b>{source}</b>: {mae:.2f}°F MAE {quality}")

        lines.extend([
            "",
            "<i>MAE = Mean Absolute Error (how far off predictions are on average)</i>",
        ])

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in /accuracy command: {e}")
        await update.message.reply_text(
            f"❌ Error fetching accuracy: {str(e)[:100]}",
            parse_mode="HTML"
        )


async def confidence_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /confidence command - shows model agreement for location (V2).

    Usage: /confidence <city>
    """
    args = context.args

    if not args:
        await update.message.reply_text(
            "❌ <b>Usage:</b> /confidence &lt;city&gt;\n\n"
            "Example:\n"
            "<code>/confidence NYC</code>",
            parse_mode="HTML"
        )
        return

    city_input = " ".join(args).strip()

    await update.message.reply_text(
        f"🔍 Analyzing pattern confidence for <b>{city_input}</b>...",
        parse_mode="HTML"
    )

    try:
        from src.analysis.patterns import classify_pattern
        from datetime import date

        # Handle common abbreviations
        city_map = {
            "NYC": "New York, NY",
            "LA": "Los Angeles, CA",
            "SF": "San Francisco, CA",
            "CHI": "Chicago, IL",
            "DC": "Washington, DC",
            "PHILLY": "Philadelphia, PA",
            "MIAMI": "Miami, FL",
            "DENVER": "Denver, CO",
        }

        city_upper = city_input.upper()
        location = city_map.get(city_upper, city_input)

        # Get today's date for analysis
        target_date = date.today().isoformat()

        result = classify_pattern(location, target_date)

        # Build confidence analysis message
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴", "VERY_LOW": "⚫"}.get(
            result.confidence_level, "⚪"
        )
        rec_emoji = {"BET": "✅", "CAUTION": "⚠️", "AVOID": "🚫"}.get(
            result.recommendation, "❓"
        )

        lines = [
            f"🎯 <b>Pattern Analysis: {result.location}</b>",
            f"📅 Target Date: {result.target_date}",
            "",
            f"<b>Pattern Type:</b> {result.pattern_type.upper()}",
            f"{conf_emoji} <b>Confidence:</b> {result.confidence_level} ({result.confidence_score:.0f}/100)",
            "",
            "<b>Model Agreement:</b>",
            f"  • Sources Used: {result.models_used}",
            f"  • High Temp Spread: {result.model_spread_high:.1f}°F",
            f"  • Low Temp Spread: {result.model_spread_low:.1f}°F",
            f"  • Agreement Score: {result.model_agreement_score:.0%}",
            "",
            "<b>Situation Flags:</b>",
        ]

        if result.is_frontal_passage:
            lines.append("  ⚠️ Frontal passage expected")
        if result.is_near_threshold:
            lines.append("  ⚠️ Near Kalshi threshold")
        if result.is_extreme_temp:
            lines.append("  ⚠️ Extreme temperatures")
        if not (result.is_frontal_passage or result.is_near_threshold or result.is_extreme_temp):
            lines.append("  ✅ No warning flags")

        lines.extend([
            "",
            f"{rec_emoji} <b>Recommendation:</b> {result.recommendation}",
            f"<i>{result.explanation}</i>",
        ])

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in /confidence command: {e}")
        await update.message.reply_text(
            f"❌ Error analyzing confidence: {str(e)[:100]}",
            parse_mode="HTML"
        )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command - shows last 10 alerts sent.

    Displays alerts from the alerted_markets table with ticker, edge%, side, and time ago.
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='alerted_markets'
        """)
        if not cursor.fetchone():
            conn.close()
            await update.message.reply_text(
                "📭 <b>No Alert History</b>\n\n"
                "No alerts have been sent yet. Start the scanner with /start.",
                parse_mode="HTML"
            )
            return

        # Get last 10 alerts
        cursor.execute("""
            SELECT ticker, alerted_at, edge_pct, side, confidence
            FROM alerted_markets
            ORDER BY alerted_at DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            await update.message.reply_text(
                "📭 <b>No Alert History</b>\n\n"
                "No alerts have been sent yet. Start the scanner with /start.",
                parse_mode="HTML"
            )
            return

        lines = [
            "📜 <b>Last 10 Alerts</b>",
            "",
        ]

        now = datetime.now()
        for row in rows:
            ticker = row["ticker"]
            alerted_at = row["alerted_at"]
            edge_pct = row["edge_pct"]
            side = row["side"]

            # Calculate time ago
            try:
                alert_time = datetime.fromisoformat(alerted_at)
                delta = now - alert_time
                if delta.days > 0:
                    time_ago = f"{delta.days}d ago"
                elif delta.seconds >= 3600:
                    hours = delta.seconds // 3600
                    time_ago = f"{hours}h ago"
                elif delta.seconds >= 60:
                    mins = delta.seconds // 60
                    time_ago = f"{mins}m ago"
                else:
                    time_ago = "just now"
            except Exception:
                time_ago = "unknown"

            # Format edge with sign
            edge_emoji = "📈" if edge_pct > 0 else "📉"
            side_emoji = "✅" if side == "YES" else "❌"

            lines.append(
                f"{edge_emoji} <code>{ticker[:20]}</code>"
            )
            lines.append(
                f"   {edge_pct:+.0f}% edge | {side_emoji} {side} | {time_ago}"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in /history command: {e}")
        await update.message.reply_text(
            f"❌ Error fetching history: {str(e)[:100]}",
            parse_mode="HTML"
        )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command - shows scanner statistics.

    Shows total scans, edges found, alerts sent, and uptime information.
    """
    try:
        # Lazy import to get scanner log path
        from src.kalshi.scheduler import SCANNER_LOG_PATH

        # Get state values
        scan_count = get_state("scan_count", "0")
        total_alerts = get_state("total_alerts", "0")
        last_started = get_state("last_started", "")
        muted = get_state("muted", "false") == "true"

        # Calculate uptime
        uptime_text = "N/A"
        if last_started:
            try:
                start_time = datetime.fromisoformat(last_started)
                now = datetime.now()
                delta = now - start_time
                days = delta.days
                hours = delta.seconds // 3600
                mins = (delta.seconds % 3600) // 60
                if days > 0:
                    uptime_text = f"{days}d {hours}h {mins}m"
                elif hours > 0:
                    uptime_text = f"{hours}h {mins}m"
                else:
                    uptime_text = f"{mins}m"
            except Exception:
                pass

        # Count alerts sent today from alerted_markets
        alerts_today = 0
        total_alerted = 0
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='alerted_markets'
            """)
            if cursor.fetchone():
                # Total alerted
                cursor.execute("SELECT COUNT(*) FROM alerted_markets")
                total_alerted = cursor.fetchone()[0]

                # Alerts today
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cursor.execute(
                    "SELECT COUNT(*) FROM alerted_markets WHERE alerted_at >= ?",
                    (today_start.isoformat(),)
                )
                alerts_today = cursor.fetchone()[0]

            conn.close()
        except Exception:
            pass

        # Read scanner log for historical stats
        total_scans_logged = 0
        total_edges_found = 0
        if SCANNER_LOG_PATH.exists():
            try:
                import csv
                with open(SCANNER_LOG_PATH, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        total_scans_logged += 1
                        total_edges_found += int(row.get("edges_found", 0))
            except Exception:
                pass

        # Build message
        muted_emoji = "🔇" if muted else "🔊"
        running = is_scanner_running()
        status_emoji = "🟢" if running else "🔴"

        lines = [
            "📊 <b>Scanner Statistics</b>",
            "",
            f"{status_emoji} <b>Status:</b> {'Running' if running else 'Stopped'}",
            f"{muted_emoji} <b>Alerts:</b> {'Muted' if muted else 'Enabled'}",
            "",
            "<b>Session Stats:</b>",
            f"   Scans this session: {scan_count}",
            f"   Alerts this session: {total_alerts}",
            f"   Uptime: {uptime_text}",
            "",
            "<b>Historical Stats:</b>",
            f"   Total scans logged: {total_scans_logged}",
            f"   Total edges found: {total_edges_found}",
            f"   Markets alerted (all time): {total_alerted}",
            f"   Alerts sent today: {alerts_today}",
        ]

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in /stats command: {e}")
        await update.message.reply_text(
            f"❌ Error fetching stats: {str(e)[:100]}",
            parse_mode="HTML"
        )


async def mute_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /mute command - temporarily disables alerts.

    The scanner continues to run but doesn't send Telegram alerts.
    """
    current_muted = get_state("muted", "false") == "true"

    if current_muted:
        await update.message.reply_text(
            "🔇 Alerts are already muted.\n\n"
            "Use /unmute to re-enable alerts.",
            parse_mode="HTML"
        )
        return

    set_state("muted", "true")

    await update.message.reply_text(
        "🔇 <b>Alerts Muted</b>\n\n"
        "The scanner will continue running but won't send alerts.\n"
        "Edge opportunities will still be logged.\n\n"
        "Use /unmute to re-enable alerts.",
        parse_mode="HTML"
    )


async def unmute_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /unmute command - re-enables alerts.

    Resumes sending Telegram alerts when edge opportunities are found.
    """
    current_muted = get_state("muted", "false") == "true"

    if not current_muted:
        await update.message.reply_text(
            "🔊 Alerts are already enabled.\n\n"
            "You'll receive notifications when edge opportunities are found.",
            parse_mode="HTML"
        )
        return

    set_state("muted", "false")

    await update.message.reply_text(
        "🔊 <b>Alerts Enabled</b>\n\n"
        "You'll now receive notifications when edge opportunities are found.\n\n"
        "Use /mute to temporarily disable alerts.",
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
# Callback query handlers for inline keyboards
# ============================================================================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all callback queries from inline keyboard buttons.

    This handler routes button presses to the appropriate action based on
    the callback_data value.
    """
    query = update.callback_query

    # Always acknowledge the callback to stop loading indicator
    await query.answer()

    data = query.data

    if data == "start_scanner":
        # Start the scanner
        if is_scanner_running():
            await query.edit_message_text(
                "🟢 Scanner is already running!\n\n"
                "Use /status to see current status.",
                parse_mode="HTML"
            )
        else:
            interval = get_state("interval", "60")
            min_edge = get_state("min_edge", "10")
            days_ahead = get_state("days_ahead", "7")

            if start_scanner():
                await query.edit_message_text(
                    "🟢 <b>Scanner Started!</b>\n\n"
                    f"Interval: {interval} min\n"
                    f"Min edge: {min_edge}%\n"
                    f"Days ahead: {days_ahead}\n\n"
                    "Use /status to check progress.",
                    parse_mode="HTML"
                )
            else:
                await query.edit_message_text(
                    "❌ Failed to start scanner.",
                    parse_mode="HTML"
                )

    elif data == "stop_scanner":
        # Stop the scanner
        if not is_scanner_running():
            await query.edit_message_text(
                "🔴 Scanner is not running.\n\n"
                "Use /start to start it.",
                parse_mode="HTML"
            )
        else:
            if stop_scanner():
                scan_count = get_state("scan_count", "0")
                total_alerts = get_state("total_alerts", "0")
                await query.edit_message_text(
                    "🔴 <b>Scanner Stopped</b>\n\n"
                    f"Total scans: {scan_count}\n"
                    f"Total alerts: {total_alerts}\n\n"
                    "Use /start to restart.",
                    parse_mode="HTML"
                )
            else:
                await query.edit_message_text(
                    "❌ Failed to stop scanner.",
                    parse_mode="HTML"
                )

    elif data == "scan_now":
        # Trigger immediate scan
        await query.edit_message_text(
            "🔍 Running scan... please wait.",
            parse_mode="HTML"
        )

        try:
            from src.kalshi.scheduler import run_single_scan

            min_edge = float(get_state("min_edge", "10"))
            max_series = int(get_state("max_series", "100"))
            days_ahead = int(get_state("days_ahead", "7"))

            result = run_single_scan(
                min_edge=min_edge,
                max_series=max_series,
                days_ahead=days_ahead,
                send_alerts=False,
                verbose=False,
            )

            set_state("last_scan", datetime.now().isoformat())
            set_state("last_edges_found", str(result["edges_found"]))

            if result["edges_found"] > 0:
                # Add button to view edges
                keyboard = [[InlineKeyboardButton("📊 View Edges", callback_data="view_edges")]]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    f"✅ <b>Scan Complete</b>\n\n"
                    f"📊 Found {result['edges_found']} edge opportunities\n"
                    f"🆕 {result['new_opportunities']} new\n"
                    f"⏱️ Duration: {result['scan_duration']:.1f}s",
                    parse_mode="HTML",
                    reply_markup=reply_markup
                )
            else:
                await query.edit_message_text(
                    f"✅ <b>Scan Complete</b>\n\n"
                    f"No edge opportunities found above {min_edge}%.\n"
                    f"⏱️ Duration: {result['scan_duration']:.1f}s",
                    parse_mode="HTML"
                )

        except Exception as e:
            logger.error(f"Error in scan callback: {e}")
            await query.edit_message_text(
                f"❌ Scan failed: {str(e)[:100]}",
                parse_mode="HTML"
            )

    elif data == "view_edges":
        # Show top edges
        try:
            from src.kalshi.edge import find_edges

            min_edge = float(get_state("min_edge", "10"))
            max_series = int(get_state("max_series", "100"))
            days_ahead = int(get_state("days_ahead", "7"))

            edges = find_edges(min_edge=min_edge, max_series=max_series, days_ahead=days_ahead)

            if not edges:
                await query.edit_message_text(
                    f"No edges found above {min_edge}% threshold.",
                    parse_mode="HTML"
                )
                return

            top_edges = edges[:5]
            lines = [
                f"🎯 <b>Top {len(top_edges)} Edge Opportunities</b>",
                "",
            ]

            for i, edge in enumerate(top_edges, 1):
                edge_emoji = "📈" if edge.edge_pct > 0 else "📉"
                side_emoji = "✅" if edge.side == "YES" else "❌"
                cond_emoji = {"temp_high": "🌡️", "temp_low": "🌡️", "rain": "🌧️", "snow": "❄️"}.get(
                    edge.market.condition_type, "🌤️"
                )

                lines.append(f"<b>{i}.</b> {cond_emoji} {edge.market.location}")
                lines.append(f"   {edge_emoji} {edge.edge_pct:+.1f}% | {side_emoji} {edge.side}")

            # Add action buttons
            keyboard = [
                [InlineKeyboardButton("🔄 Scan Again", callback_data="scan_now")],
                [InlineKeyboardButton("⚙️ Settings", callback_data="show_settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "\n".join(lines),
                parse_mode="HTML",
                reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in view_edges callback: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)[:100]}", parse_mode="HTML")

    elif data == "show_settings":
        # Show settings
        interval = get_state("interval", "60")
        min_edge = get_state("min_edge", "10")
        days_ahead = get_state("days_ahead", "7")
        max_series = get_state("max_series", "100")

        await query.edit_message_text(
            f"⚙️ <b>Current Settings</b>\n\n"
            f"Edge Threshold: <b>{min_edge}%</b>\n"
            f"Scan Interval: <b>{interval} min</b>\n"
            f"Days Ahead: <b>{days_ahead}</b>\n"
            f"Max Series: <b>{max_series}</b>\n\n"
            f"Use /set to modify settings.",
            parse_mode="HTML"
        )

    elif data == "mute_alerts":
        # Mute alerts
        set_state("muted", "true")
        await query.edit_message_text(
            "🔇 <b>Alerts Muted</b>\n\n"
            "Scanner will continue but won't send notifications.\n\n"
            "Use /unmute to re-enable.",
            parse_mode="HTML"
        )

    elif data == "unmute_alerts":
        # Unmute alerts
        set_state("muted", "false")
        await query.edit_message_text(
            "🔊 <b>Alerts Enabled</b>\n\n"
            "You'll receive notifications for edge opportunities.",
            parse_mode="HTML"
        )

    elif data == "refresh_status":
        # Refresh status display
        running = is_scanner_running()
        muted = get_state("muted", "false") == "true"
        status_emoji = "🟢" if running else "🔴"

        interval = get_state("interval", "60")
        min_edge = get_state("min_edge", "10")
        scan_count = get_state("scan_count", "0")
        total_alerts = get_state("total_alerts", "0")
        last_edges = get_state("last_edges_found", "0")

        lines = [
            f"{status_emoji} <b>Status: {'Running' if running else 'Stopped'}</b>",
            f"{'🔇 Muted' if muted else '🔊 Active'}",
            "",
            f"Interval: {interval} min | Edge: {min_edge}%",
            f"Scans: {scan_count} | Alerts: {total_alerts}",
            f"Last edges: {last_edges}",
        ]

        # Build appropriate buttons based on state
        buttons = []
        if running:
            buttons.append([InlineKeyboardButton("🛑 Stop", callback_data="stop_scanner")])
        else:
            buttons.append([InlineKeyboardButton("▶️ Start", callback_data="start_scanner")])

        buttons.append([
            InlineKeyboardButton("🔍 Scan Now", callback_data="scan_now"),
            InlineKeyboardButton("🔇 Mute" if not muted else "🔊 Unmute",
                               callback_data="mute_alerts" if not muted else "unmute_alerts")
        ])
        buttons.append([InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status")])

        reply_markup = InlineKeyboardMarkup(buttons)

        await query.edit_message_text(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup
        )

    else:
        # Unknown callback
        await query.edit_message_text(
            f"Unknown action: {data}",
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


async def _send_message_async(
    bot: Bot,
    chat_id: str,
    text: str,
    parse_mode: str = "HTML",
    reply_markup: Optional[InlineKeyboardMarkup] = None
) -> bool:
    """Send message asynchronously with retry logic.

    Args:
        bot: Telegram Bot instance
        chat_id: Chat ID to send to
        text: Message text
        parse_mode: Message parse mode ("HTML" or "Markdown")
        reply_markup: Optional inline keyboard markup

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
                reply_markup=reply_markup,
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
    ]

    message = "\n".join(lines)

    # Create inline keyboard with View on Kalshi button
    keyboard = [[InlineKeyboardButton("🔗 View on Kalshi", url=market_link)]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    _rate_limit()

    try:
        return asyncio.run(_send_message_async(
            bot, TELEGRAM_CHAT_ID, message,
            parse_mode="HTML",
            reply_markup=reply_markup
        ))
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

    # Settings commands
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("set", set_command))

    # Lookup commands
    application.add_handler(CommandHandler("weather", weather_command))
    application.add_handler(CommandHandler("market", market_command))
    application.add_handler(CommandHandler("edges", edges_command))

    # V2 commands
    application.add_handler(CommandHandler("forecast", forecast_command))
    application.add_handler(CommandHandler("accuracy", accuracy_command))
    application.add_handler(CommandHandler("confidence", confidence_command))

    # Tracking commands
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("mute", mute_command))
    application.add_handler(CommandHandler("unmute", unmute_command))

    # Callback query handler for inline keyboard buttons
    application.add_handler(CallbackQueryHandler(button_callback))

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
