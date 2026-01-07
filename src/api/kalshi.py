"""Kalshi API client for prediction market data.

This module provides a client for interacting with the Kalshi trading API
to fetch weather-related prediction markets.
"""

import base64
import datetime
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.config import (
    KALSHI_API_BASE_URL,
    KALSHI_CLIENT_ID,
    KALSHI_CLIENT_SECRET,
    PROJECT_ROOT,
)


@dataclass
class Market:
    """Represents a Kalshi prediction market."""

    ticker: str
    title: str
    subtitle: str
    status: str
    yes_bid: float  # in dollars
    yes_ask: float  # in dollars
    no_bid: float  # in dollars
    no_ask: float  # in dollars
    last_price: float  # in dollars
    volume: int
    volume_24h: int
    open_interest: int
    close_time: Optional[datetime.datetime] = None
    expiration_time: Optional[datetime.datetime] = None
    event_ticker: str = ""
    rules_primary: str = ""


@dataclass
class OrderBookLevel:
    """Represents a single level in the order book."""

    price: float  # in dollars
    quantity: int


@dataclass
class OrderBook:
    """Represents the order book for a market."""

    ticker: str
    yes_bids: list[OrderBookLevel] = field(default_factory=list)
    no_bids: list[OrderBookLevel] = field(default_factory=list)


class KalshiClient:
    """Client for interacting with the Kalshi API.

    Supports authentication via RSA key signing (v2 API).
    """

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the Kalshi client.

        Args:
            api_key_id: The Kalshi API key ID. Defaults to KALSHI_CLIENT_ID from config.
            private_key_path: Path to the private key file (.key or .pem).
            private_key_pem: Private key content as string.
            base_url: Base URL for the API. Defaults to KALSHI_API_BASE_URL from config.
        """
        self.api_key_id = api_key_id or KALSHI_CLIENT_ID
        self.base_url = base_url or KALSHI_API_BASE_URL
        self._private_key = None
        self._session = requests.Session()
        self._last_request_time = 0.0

        # Try to load private key
        if private_key_pem:
            self._load_private_key_from_string(private_key_pem)
        elif private_key_path:
            self._load_private_key_from_file(private_key_path)
        else:
            # Try default locations
            self._try_load_default_private_key()

    def _try_load_default_private_key(self) -> None:
        """Try to load private key from default locations."""
        # Check if KALSHI_CLIENT_SECRET is a key or password
        # If it starts with "-----BEGIN", it's a PEM key
        if KALSHI_CLIENT_SECRET and KALSHI_CLIENT_SECRET.startswith("-----BEGIN"):
            self._load_private_key_from_string(KALSHI_CLIENT_SECRET)
            return

        # Try common locations
        possible_paths = [
            PROJECT_ROOT / "kalshi-key.key",
            PROJECT_ROOT / "private_key.pem",
            Path.home() / ".kalshi" / "private_key.pem",
            Path.home() / ".kalshi-key.key",
        ]

        for path in possible_paths:
            if path.exists():
                self._load_private_key_from_file(str(path))
                return

    def _load_private_key_from_file(self, path: str) -> None:
        """Load private key from file."""
        try:
            with open(path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
        except Exception as e:
            print(f"Warning: Could not load private key from {path}: {e}")

    def _load_private_key_from_string(self, pem_string: str) -> None:
        """Load private key from PEM string."""
        try:
            self._private_key = serialization.load_pem_private_key(
                pem_string.encode(),
                password=None,
                backend=default_backend()
            )
        except Exception as e:
            print(f"Warning: Could not load private key from string: {e}")

    def _create_signature(self, timestamp: str, method: str, path: str) -> str:
        """Create RSA-PSS signature for request authentication.

        Args:
            timestamp: Request timestamp in milliseconds.
            method: HTTP method (GET, POST, etc.).
            path: API path (without query parameters).

        Returns:
            Base64-encoded signature string.
        """
        if not self._private_key:
            raise ValueError(
                "Private key not loaded. Please provide a private key file or "
                "set KALSHI_CLIENT_SECRET to your PEM key content."
            )

        # Strip query parameters from path
        path_without_query = path.split("?")[0]

        # Build message to sign: timestamp + method + path
        message = f"{timestamp}{method}{path_without_query}".encode("utf-8")

        # Sign with RSA-PSS
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode("utf-8")

    def _get_auth_headers(self, method: str, path: str) -> dict:
        """Get authentication headers for a request.

        Args:
            method: HTTP method.
            path: API path.

        Returns:
            Dictionary of authentication headers.
        """
        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        signature = self._create_signature(timestamp, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:  # Max 1 request per second to avoid 429s
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        require_auth: bool = False,
    ) -> dict:
        """Make a request to the Kalshi API.

        Args:
            method: HTTP method.
            path: API path (relative to base URL).
            params: Query parameters.
            data: Request body data.
            require_auth: If True, require authentication. If False, use auth if available.

        Returns:
            Response data as dictionary.

        Raises:
            requests.HTTPError: If the request fails.
            ValueError: If authentication is required but not available.
        """
        self._rate_limit()

        full_path = f"/trade-api/v2{path}"
        url = f"{self.base_url.rstrip('/trade-api/v2')}{full_path}"

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Add auth headers if private key is available
        if self._private_key:
            # Build path with query params for signing
            if params:
                query_string = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
                sign_path = f"{full_path}?{query_string}" if query_string else full_path
            else:
                sign_path = full_path

            headers.update(self._get_auth_headers(method, sign_path))
        elif require_auth:
            raise ValueError(
                "This endpoint requires authentication. Please provide a private key."
            )

        # Retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=30,
            )

            if response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2, 4, 8 seconds
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        # If we exhausted retries, raise the last error
        response.raise_for_status()
        return response.json()

    def login(self) -> bool:
        """Verify API credentials are valid by making a test request.

        Returns:
            True if authentication is successful.

        Raises:
            ValueError: If private key is not loaded or authentication fails.
            requests.HTTPError: If the request fails for other reasons.
        """
        if not self._private_key:
            raise ValueError(
                "Private key not loaded. Cannot authenticate without a private key. "
                "Please:\n"
                "1. Generate an API key at https://kalshi.com/account/api-keys\n"
                "2. Save the private key to 'kalshi-key.key' in the project root, or\n"
                "3. Set KALSHI_CLIENT_SECRET to the PEM key content in .env"
            )

        # Test authentication by getting account balance
        try:
            self._request("GET", "/portfolio/balance", require_auth=True)
            return True
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please verify your API key ID and private key."
                ) from e
            raise

    def is_authenticated(self) -> bool:
        """Check if the client has authentication credentials loaded.

        Returns:
            True if a private key is loaded and ready for authenticated requests.
        """
        return self._private_key is not None

    def get_weather_markets(self, max_series: int = 100) -> list[Market]:
        """Get all weather-related prediction markets.

        Fetches markets from known weather series.

        Args:
            max_series: Maximum number of series to query (default 100).

        Returns:
            List of Market objects for weather markets.
        """
        weather_markets = []
        seen_tickers = set()

        # Weather-related series tickers (discovered from Kalshi API)
        # Note: Most active weather series use the "KX" prefix
        weather_series = [
            # NYC Weather
            "KXHIGHNY", "KXLOWNY", "KXMINNYC", "KXSNOWNY", "KXRAINNY",
            "KXNYCSNOWM", "KXRAINNYCM", "KXSNOWNYM", "KXNYCSNOWXMAS",
            "KXHIGHNY0", "KXLOWNYC", "KXLOWTNYC",
            # Chicago Weather
            "KXHIGHCHI", "KXLOWCHI", "KXCHISNOWM", "KXCHISNOWXMAS",
            "KXSNOWCHIM", "KXLOWTCHI", "KXRAINCHIM",
            # Other Cities
            "KXHIGHDEN", "KXLOWDEN", "KXLOWTDEN", "KXDENSNOWM", "KXDENSNOWXMAS",
            "KXHIGHMIA", "KXLOWMIA", "KXLOWTMIA", "KXMIASNOWM",
            "KXHIGHHOU", "KXHIGHOU", "KXHOUHIGH", "KXRAINHOU", "KXRAINHOUM",
            "KXHIGHLAX", "KXLOWLAX", "KXLOWTLAX", "KXLAXSNOWM", "KXRAINLAXM",
            "KXHIGHPHIL", "KXLOWPHIL", "KXLOWTPHIL", "KXPHILSNOWM", "KXPHILHIGH",
            "KXHIGHAUS", "KXLOWAUS", "KXLOWTAUS", "KXAUSSNOWM", "KXRAINAUSM",
            "KXBOSSNOWM", "KXBOSSNOWXMAS",
            "KXSFOSNOWM", "KXRAINSEA", "KXSEASNOWM", "KXRAINSEAM",
            "KXDCSNOWM", "KXDALSNOWM", "KXRAINDALM",
            "KXHOUSNOWM", "KXDETSNOWM", "KXSLCSNOWM",
            "KXJACWSNOWM",  # Jackson, WY
            # Death Valley
            "KXDVHIGH",
            # Regional/National
            "KXHIGHUS", "KXAVGTEMP", "KXTEMP", "KXTEMPMON",
            "KXGTEMP", "KXHOTYEAR", "KXHMONTHRANGE",
            "KXMAXTEMP100", "KXCITIESWEATHER",
            # Hurricanes
            "KXHURNYC", "KXHURTB", "KXHURORL", "KXHURMIA", "KXHURNO",
            "KXHURJACKFL", "KXHURCHARL", "KXHURSAV", "KXHURNOR",
            "KXHURWIL", "KXHURMYR", "KXHURHAT",
            "KXHURNJ", "KXHURCAL", "KXHURCOASTTEX",
            "KXHURCTOT", "KXHURCMAJ", "KXHURCTOTMAJ",
            "KXTROPSTORM", "KXHURCLAND",
            "KXHURCAT", "KXHURCATH", "KXHURCATI", "KXHURCATFL",
            "KXNEXTHURDATE", "KXNEXTCAT5HURDATE",
            "KXHURPATHFLA", "KXHURPATHHOU", "KXHURPATHGULFCOAST",
            "KXHURPATHGENERALMAJOR", "KXHURPATHGENERAL", "KXHURPATHHAWAII",
            "KXHURPATHSCAROLINA",
            # Rain
            "KXRAINNYC", "KXRAINNO", "KXRAINDENM", "KXRAINSFOM", "KXRAINMIAM",
            "KXRAINNOSB",  # Super Bowl Sunday rain
            # Snow
            "KXSNOWAZ", "KXSNOWS",  # White Christmas
            # Other
            "KXMICHTEMP",  # Lake Michigan
            "KXGSTORM",  # Geomagnetic storm
            "KXHEATWARNING", "KXNYCHOT",
            # Legacy series (without KX prefix)
            "HIGHNY", "LOWNY", "MINNYC", "SNOWNY", "RAINNY",
            "HIGHCHI", "LOWCHI", "SNOWCHIM",
            "HIGHUS", "AVGTEMP", "TEMP", "TEMPMON",
            "GTEMP", "HOTYEAR",
            "HURNYC", "HURTB", "HURMIA", "HURNO",
            "HURCTOT", "HURCMAJ", "HURCLAND",
            "TROPSTORM", "HURCAT",
            "RAINHOU", "RAINSEA", "RAINNY", "RAINMIA", "RAINNO",
            "SNOW", "SNOWNYM",
        ]

        # Query each weather series for open markets
        queried = 0
        for series in weather_series:
            if queried >= max_series:
                break

            try:
                params = {
                    "series_ticker": series,
                    "status": "open",
                    "limit": 100,
                }
                response = self._request("GET", "/markets", params=params)
                markets = response.get("markets", [])

                for m in markets:
                    ticker = m.get("ticker", "")
                    if ticker not in seen_tickers:
                        weather_markets.append(self._parse_market(m))
                        seen_tickers.add(ticker)

                queried += 1

            except Exception:
                # Series may not exist or have no open markets
                continue

        return weather_markets

    def get_market_details(self, ticker: str) -> Market:
        """Get detailed information about a specific market.

        Args:
            ticker: The market ticker (e.g., "WEATHER-NYC-RAIN").

        Returns:
            Market object with full details.
        """
        response = self._request("GET", f"/markets/{ticker}")
        market_data = response.get("market", {})
        return self._parse_market(market_data)

    def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """Get the order book for a market.

        Args:
            ticker: The market ticker.
            depth: Number of price levels to retrieve (0-100, 0 = all).

        Returns:
            OrderBook object with bid/ask data.
        """
        params = {"depth": depth} if depth > 0 else {}
        response = self._request("GET", f"/markets/{ticker}/orderbook", params=params)
        orderbook_data = response.get("orderbook", {})

        yes_bids = []
        no_bids = []

        # Parse yes bids (price in dollars format)
        for level in orderbook_data.get("yes_dollars", []):
            if len(level) >= 2:
                price = float(level[0])
                quantity = int(level[1])
                yes_bids.append(OrderBookLevel(price=price, quantity=quantity))

        # Parse no bids
        for level in orderbook_data.get("no_dollars", []):
            if len(level) >= 2:
                price = float(level[0])
                quantity = int(level[1])
                no_bids.append(OrderBookLevel(price=price, quantity=quantity))

        return OrderBook(ticker=ticker, yes_bids=yes_bids, no_bids=no_bids)

    def _parse_market(self, data: dict) -> Market:
        """Parse market data from API response.

        Args:
            data: Raw market data from API.

        Returns:
            Market object.
        """
        def parse_price(val) -> float:
            """Parse price value (handles string or number)."""
            if val is None:
                return 0.0
            if isinstance(val, str):
                try:
                    return float(val)
                except ValueError:
                    return 0.0
            return float(val)

        def parse_datetime(val) -> Optional[datetime.datetime]:
            """Parse datetime from ISO string."""
            if not val:
                return None
            try:
                # Handle Z suffix
                if val.endswith("Z"):
                    val = val[:-1] + "+00:00"
                return datetime.datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return None

        return Market(
            ticker=data.get("ticker", ""),
            title=data.get("title", "") or data.get("yes_sub_title", ""),
            subtitle=data.get("subtitle", "") or data.get("no_sub_title", ""),
            status=data.get("status", ""),
            yes_bid=parse_price(data.get("yes_bid_dollars")),
            yes_ask=parse_price(data.get("yes_ask_dollars")),
            no_bid=parse_price(data.get("no_bid_dollars")),
            no_ask=parse_price(data.get("no_ask_dollars")),
            last_price=parse_price(data.get("last_price_dollars")),
            volume=data.get("volume", 0) or 0,
            volume_24h=data.get("volume_24h", 0) or 0,
            open_interest=data.get("open_interest", 0) or 0,
            close_time=parse_datetime(data.get("close_time")),
            expiration_time=parse_datetime(
                data.get("expected_expiration_time") or data.get("latest_expiration_time")
            ),
            event_ticker=data.get("event_ticker", ""),
            rules_primary=data.get("rules_primary", "") or "",
        )


def get_kalshi_client(
    api_key_id: Optional[str] = None,
    private_key_path: Optional[str] = None,
) -> KalshiClient:
    """Create and return a configured Kalshi client.

    Args:
        api_key_id: Optional API key ID override.
        private_key_path: Optional path to private key file.

    Returns:
        Configured KalshiClient instance.
    """
    return KalshiClient(
        api_key_id=api_key_id,
        private_key_path=private_key_path,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing Kalshi API client...")

    client = KalshiClient()

    try:
        # Try to get weather markets
        print("\nFetching weather markets...")
        markets = client.get_weather_markets()
        print(f"Found {len(markets)} weather-related markets")

        for market in markets[:5]:
            print(f"\n  {market.ticker}")
            print(f"    Title: {market.title}")
            print(f"    Yes: ${market.yes_bid:.2f} / ${market.yes_ask:.2f}")
            print(f"    Volume: {market.volume}")

        if markets:
            # Get orderbook for first market
            first_ticker = markets[0].ticker
            print(f"\nOrderbook for {first_ticker}:")
            orderbook = client.get_orderbook(first_ticker)
            print(f"  Yes bids: {len(orderbook.yes_bids)} levels")
            print(f"  No bids: {len(orderbook.no_bids)} levels")

    except ValueError as e:
        print(f"\nAuthentication error: {e}")
    except requests.HTTPError as e:
        print(f"\nAPI error: {e}")
