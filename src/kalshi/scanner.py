"""Weather market scanner for Kalshi prediction markets.

This module scans Kalshi markets to find weather-related prediction markets,
parses their titles to extract location, date, and condition information,
and maps them to geographic coordinates for model predictions.
"""

import datetime
import re
from dataclasses import dataclass
from typing import Optional

from src.api.kalshi import KalshiClient, Market, get_kalshi_client
from src.api.geocoding import get_coordinates


@dataclass
class WeatherMarket:
    """Parsed weather market information from Kalshi.

    Attributes:
        ticker: Kalshi market ticker (e.g., "KXHIGHNY-26JAN09-35")
        title: Full market title from Kalshi
        location: Parsed location name (e.g., "New York")
        state: State abbreviation if available (e.g., "NY")
        lat: Latitude of the location
        lon: Longitude of the location
        target_date: Date the market is about
        condition_type: Type of weather condition (temp_high, temp_low, rain, snow)
        threshold: Threshold value (e.g., 35 for "above 35F")
        comparison: Comparison operator (above, below, at_least, at_most)
        yes_bid: Current yes bid price (probability)
        yes_ask: Current yes ask price
        last_price: Last traded price
        volume: Total trading volume
        expiration_time: When the market expires/settles
    """
    ticker: str
    title: str
    location: str
    state: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    target_date: Optional[datetime.date]
    condition_type: str  # temp_high, temp_low, rain, snow, hurricane
    threshold: Optional[float]
    comparison: Optional[str]  # above, below, at_least, at_most
    yes_bid: float
    yes_ask: float
    last_price: float
    volume: int
    expiration_time: Optional[datetime.datetime]


# Location mapping from Kalshi ticker abbreviations to city/state
KALSHI_LOCATION_MAP = {
    # Major cities from ticker prefixes
    "NY": ("New York", "NY"),
    "NYC": ("New York", "NY"),
    "CHI": ("Chicago", "IL"),
    "LA": ("Los Angeles", "CA"),
    "LAX": ("Los Angeles", "CA"),
    "MIA": ("Miami", "FL"),
    "HOU": ("Houston", "TX"),
    "DEN": ("Denver", "CO"),
    "PHI": ("Philadelphia", "PA"),
    "PHIL": ("Philadelphia", "PA"),
    "AUS": ("Austin", "TX"),
    "BOS": ("Boston", "MA"),
    "SFO": ("San Francisco", "CA"),
    "SF": ("San Francisco", "CA"),
    "SEA": ("Seattle", "WA"),
    "DAL": ("Dallas", "TX"),
    "DC": ("Washington", "DC"),
    "DET": ("Detroit", "MI"),
    "SLC": ("Salt Lake City", "UT"),
    "JACW": ("Jackson", "WY"),
    "DV": ("Death Valley", "CA"),
    "TB": ("Tampa Bay", "FL"),
    "ORL": ("Orlando", "FL"),
    "NO": ("New Orleans", "LA"),
    "JACKFL": ("Jacksonville", "FL"),
    "CHARL": ("Charleston", "SC"),
    "SAV": ("Savannah", "GA"),
    "NOR": ("Norfolk", "VA"),
    "WIL": ("Wilmington", "NC"),
    "MYR": ("Myrtle Beach", "SC"),
    "HAT": ("Hatteras", "NC"),
    "NJ": ("New Jersey", "NJ"),
}


def _extract_location_from_ticker(ticker: str) -> tuple[Optional[str], Optional[str]]:
    """Extract location from Kalshi ticker.

    Kalshi weather tickers usually follow patterns like:
    - KXHIGHNY-26JAN09-35 (high temp in NY)
    - KXSNOWCHI-26JAN09-0.5 (snow in Chicago)

    Args:
        ticker: Kalshi market ticker

    Returns:
        Tuple of (city_name, state_abbrev) or (None, None) if not found
    """
    # Remove common prefixes
    ticker_upper = ticker.upper()

    # Try to extract location code from ticker
    # Pattern: KX<CONDITION><LOCATION>- or KX<LOCATION><CONDITION>-
    for loc_code, (city, state) in KALSHI_LOCATION_MAP.items():
        if loc_code in ticker_upper:
            return city, state

    return None, None


def _extract_location_from_title(title: str) -> tuple[Optional[str], Optional[str]]:
    """Extract location from market title.

    Titles often include location names like:
    - "NYC high temperature above 35F"
    - "Will it rain in New York?"
    - "Chicago daily snowfall"

    Args:
        title: Market title string

    Returns:
        Tuple of (city_name, state_abbrev) or (None, None) if not found
    """
    title_lower = title.lower()

    # Check for known location keywords
    location_keywords = {
        "nyc": ("New York", "NY"),
        "new york": ("New York", "NY"),
        "chicago": ("Chicago", "IL"),
        "los angeles": ("Los Angeles", "CA"),
        "la ": ("Los Angeles", "CA"),
        "miami": ("Miami", "FL"),
        "houston": ("Houston", "TX"),
        "denver": ("Denver", "CO"),
        "philadelphia": ("Philadelphia", "PA"),
        "austin": ("Austin", "TX"),
        "boston": ("Boston", "MA"),
        "san francisco": ("San Francisco", "CA"),
        "seattle": ("Seattle", "WA"),
        "dallas": ("Dallas", "TX"),
        "washington": ("Washington", "DC"),
        "d.c.": ("Washington", "DC"),
        "detroit": ("Detroit", "MI"),
        "salt lake": ("Salt Lake City", "UT"),
        "jackson": ("Jackson", "WY"),
        "death valley": ("Death Valley", "CA"),
        "tampa": ("Tampa Bay", "FL"),
        "orlando": ("Orlando", "FL"),
        "new orleans": ("New Orleans", "LA"),
    }

    for keyword, (city, state) in location_keywords.items():
        if keyword in title_lower:
            return city, state

    return None, None


def _parse_condition_type(ticker: str, title: str) -> str:
    """Parse the type of weather condition from ticker/title.

    Args:
        ticker: Market ticker
        title: Market title

    Returns:
        Condition type string: temp_high, temp_low, rain, snow, hurricane
    """
    ticker_upper = ticker.upper()
    title_lower = title.lower()

    # Check ticker patterns
    if "HIGH" in ticker_upper or "high" in title_lower:
        return "temp_high"
    if "LOW" in ticker_upper or "low" in title_lower:
        return "temp_low"
    if "SNOW" in ticker_upper or "snow" in title_lower:
        return "snow"
    if "RAIN" in ticker_upper or "rain" in title_lower or "precip" in title_lower:
        return "rain"
    if "HUR" in ticker_upper or "hurricane" in title_lower:
        return "hurricane"
    if "TEMP" in ticker_upper or "temperature" in title_lower:
        return "temp_high"  # Default to high temp

    return "unknown"


def _parse_threshold(ticker: str, title: str) -> tuple[Optional[float], Optional[str]]:
    """Parse threshold value and comparison from ticker/title.

    Examples:
    - "above 35F" -> (35.0, "above")
    - "at least 0.1 inches" -> (0.1, "at_least")
    - "KXHIGHNY-26JAN09-35" -> (35.0, "above")

    Args:
        ticker: Market ticker
        title: Market title

    Returns:
        Tuple of (threshold_value, comparison_operator)
    """
    # Try to extract from ticker (e.g., -35 at the end)
    ticker_match = re.search(r'-(\d+\.?\d*)$', ticker)
    if ticker_match:
        threshold = float(ticker_match.group(1))
        # Determine comparison from context
        ticker_upper = ticker.upper()
        if "HIGH" in ticker_upper or "MAX" in ticker_upper:
            return threshold, "above"
        elif "LOW" in ticker_upper or "MIN" in ticker_upper:
            return threshold, "below"
        return threshold, "at_least"

    # Try to extract from title
    title_lower = title.lower()

    # Pattern: "above/below/at least/at most X"
    above_match = re.search(r'above\s+(\d+\.?\d*)', title_lower)
    if above_match:
        return float(above_match.group(1)), "above"

    below_match = re.search(r'below\s+(\d+\.?\d*)', title_lower)
    if below_match:
        return float(below_match.group(1)), "below"

    at_least_match = re.search(r'at\s+least\s+(\d+\.?\d*)', title_lower)
    if at_least_match:
        return float(at_least_match.group(1)), "at_least"

    at_most_match = re.search(r'at\s+most\s+(\d+\.?\d*)', title_lower)
    if at_most_match:
        return float(at_most_match.group(1)), "at_most"

    # Pattern: >X° or <X° (common in Kalshi titles)
    greater_match = re.search(r'>\s*(\d+\.?\d*)\s*°', title)
    if greater_match:
        return float(greater_match.group(1)), "above"

    less_match = re.search(r'<\s*(\d+\.?\d*)\s*°', title)
    if less_match:
        return float(less_match.group(1)), "below"

    # Pattern: just a number with F or degrees
    temp_match = re.search(r'(\d+\.?\d*)\s*[°fF]', title)
    if temp_match:
        return float(temp_match.group(1)), "above"

    return None, None


def _parse_target_date(ticker: str, title: str, expiration: Optional[datetime.datetime]) -> Optional[datetime.date]:
    """Parse the target date for the weather event.

    Kalshi tickers often include date like:
    - KXHIGHNY-26JAN09 (January 9, 2026) - format is YYMMMDD

    Args:
        ticker: Market ticker
        title: Market title
        expiration: Market expiration time

    Returns:
        Target date or None if not determinable
    """
    # Try to extract from ticker (format: -YYMMMDD)
    # Example: -26JAN07 means year 2026, January 7th
    date_match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})(?:-|$)', ticker.upper())
    if date_match:
        year_short = int(date_match.group(1))
        month_str = date_match.group(2)
        day = int(date_match.group(3))

        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        month = month_map.get(month_str, 1)

        # Year format is always YY (e.g., 26 = 2026)
        year = 2000 + year_short

        try:
            return datetime.date(year, month, day)
        except ValueError:
            pass

    # Fall back to expiration date
    if expiration:
        return expiration.date()

    return None


def _parse_market(market: Market) -> WeatherMarket:
    """Parse a Kalshi Market into a WeatherMarket with extracted info.

    Args:
        market: Raw Kalshi Market object

    Returns:
        WeatherMarket with parsed location, condition, and threshold info
    """
    # Extract location from ticker first, then title
    city, state = _extract_location_from_ticker(market.ticker)
    if not city:
        city, state = _extract_location_from_title(market.title)

    # Get coordinates
    lat, lon = None, None
    if city:
        coords = get_coordinates(city, state)
        if coords:
            lat, lon = coords

    # Parse condition type
    condition_type = _parse_condition_type(market.ticker, market.title)

    # Parse threshold
    threshold, comparison = _parse_threshold(market.ticker, market.title)

    # Parse target date
    target_date = _parse_target_date(market.ticker, market.title, market.expiration_time)

    return WeatherMarket(
        ticker=market.ticker,
        title=market.title,
        location=city or "Unknown",
        state=state,
        lat=lat,
        lon=lon,
        target_date=target_date,
        condition_type=condition_type,
        threshold=threshold,
        comparison=comparison,
        yes_bid=market.yes_bid,
        yes_ask=market.yes_ask,
        last_price=market.last_price,
        volume=market.volume,
        expiration_time=market.expiration_time,
    )


def scan_weather_markets(
    max_series: int = 100,
    days_ahead: int = 7,
    client: Optional[KalshiClient] = None,
) -> list[WeatherMarket]:
    """Scan Kalshi for active weather prediction markets.

    Finds all open weather markets, parses their information, and filters
    to those expiring within the specified number of days.

    Args:
        max_series: Maximum number of series to query (default 100)
        days_ahead: Only include markets expiring within this many days (default 7)
        client: Optional KalshiClient instance. If not provided, creates a new one.

    Returns:
        List of WeatherMarket objects, sorted by expiration time

    Example:
        >>> markets = scan_weather_markets()
        >>> for m in markets:
        ...     print(f"{m.ticker}: {m.location} - {m.condition_type}")
    """
    if client is None:
        client = get_kalshi_client()

    # Get all weather markets from Kalshi
    raw_markets = client.get_weather_markets(max_series=max_series)

    # Parse and filter markets
    parsed_markets = []
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=days_ahead)

    for market in raw_markets:
        # Skip if no expiration or expired
        if not market.expiration_time:
            continue

        # Make expiration timezone-aware if needed
        exp_time = market.expiration_time
        if exp_time.tzinfo is None:
            exp_time = exp_time.replace(tzinfo=datetime.timezone.utc)

        # Filter to markets expiring within days_ahead
        if exp_time > cutoff_date:
            continue

        # Skip already expired markets
        now = datetime.datetime.now(datetime.timezone.utc)
        if exp_time < now:
            continue

        parsed = _parse_market(market)
        parsed_markets.append(parsed)

    # Sort by expiration time
    parsed_markets.sort(key=lambda m: m.expiration_time or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc))

    return parsed_markets


def format_weather_market(market: WeatherMarket) -> str:
    """Format a WeatherMarket for display.

    Args:
        market: WeatherMarket to format

    Returns:
        Formatted string representation
    """
    lines = [
        f"Market: {market.ticker}",
        f"  Title: {market.title}",
        f"  Location: {market.location}, {market.state or 'N/A'}",
        f"  Coordinates: ({market.lat}, {market.lon})" if market.lat else "  Coordinates: N/A",
        f"  Condition: {market.condition_type}",
        f"  Threshold: {market.threshold} ({market.comparison})" if market.threshold else "  Threshold: N/A",
        f"  Target Date: {market.target_date}" if market.target_date else "  Target Date: N/A",
        f"  Yes Bid/Ask: ${market.yes_bid:.2f} / ${market.yes_ask:.2f}",
        f"  Last Price: ${market.last_price:.2f}",
        f"  Volume: {market.volume}",
        f"  Expires: {market.expiration_time}" if market.expiration_time else "  Expires: N/A",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("Scanning weather markets from Kalshi...")
    print("=" * 60)

    markets = scan_weather_markets()

    print(f"\nFound {len(markets)} weather markets expiring in the next 7 days:\n")

    for market in markets[:10]:  # Show first 10
        print(format_weather_market(market))
        print("-" * 40)

    if len(markets) > 10:
        print(f"\n... and {len(markets) - 10} more markets")
