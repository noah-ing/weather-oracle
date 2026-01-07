"""Telegram bot for sending Weather Oracle alerts.

This module provides functions to send alerts and edge opportunity notifications
via Telegram to a configured chat.

Usage:
    >>> from src.telegram.bot import send_alert, send_edge_alert
    >>> send_alert("Test message from Weather Oracle!")
    >>> # Or send formatted edge alerts
    >>> from src.kalshi.edge import find_edges
    >>> edges = find_edges(min_edge=15)
    >>> for edge in edges:
    ...     send_edge_alert(edge)
"""

import asyncio
import time
from typing import Optional

from telegram import Bot
from telegram.error import (
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.kalshi.edge import EdgeOpportunity


# Rate limiting
_last_message_time: float = 0.0
MIN_MESSAGE_INTERVAL = 1.0  # Minimum 1 second between messages


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
            print(f"Telegram error: {e}")
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


if __name__ == "__main__":
    # Test sending an alert
    print("Testing Telegram bot...")

    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set in .env")
    elif not TELEGRAM_CHAT_ID:
        print("TELEGRAM_CHAT_ID not set in .env")
    else:
        print(f"Bot token configured: {TELEGRAM_BOT_TOKEN[:10]}...")
        print(f"Chat ID configured: {TELEGRAM_CHAT_ID}")

        success = send_alert("🌤️ Test from Weather Oracle!\n\nThis is a test message to verify the Telegram integration is working.")

        if success:
            print("✅ Test message sent successfully!")
        else:
            print("❌ Failed to send test message")
