"""US city geocoding lookup using Open-Meteo Geocoding API."""

import time
from dataclasses import dataclass
from typing import Optional, Tuple
import requests

from src.config import OPEN_METEO_GEOCODING_URL, API_RATE_LIMIT_SECONDS


# In-memory cache for geocoding results
_geocoding_cache: dict[str, Optional[Tuple[float, float]]] = {}

# Rate limiting
_last_request_time: float = 0.0


@dataclass
class GeocodingResult:
    """Result from geocoding lookup."""
    name: str
    latitude: float
    longitude: float
    country: str
    admin1: Optional[str] = None  # State/province
    population: Optional[int] = None


def _rate_limit() -> None:
    """Enforce rate limiting between API requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < API_RATE_LIMIT_SECONDS:
        time.sleep(API_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _make_cache_key(city_name: str, state: Optional[str] = None) -> str:
    """Create a cache key from city name and optional state."""
    key = city_name.lower().strip()
    if state:
        key = f"{key},{state.lower().strip()}"
    return key


def get_coordinates(
    city_name: str,
    state: Optional[str] = None
) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude for a US city.

    Args:
        city_name: Name of the city (e.g., "New York", "Los Angeles")
        state: Optional state abbreviation or name (e.g., "NY", "California")

    Returns:
        Tuple of (latitude, longitude) or None if city not found.

    Example:
        >>> get_coordinates("New York", "NY")
        (40.7143, -74.006)
    """
    # Check cache first
    cache_key = _make_cache_key(city_name, state)
    if cache_key in _geocoding_cache:
        return _geocoding_cache[cache_key]

    # Make API request with rate limiting
    # Note: Search by city name only, then filter by state
    _rate_limit()

    try:
        params = {
            "name": city_name,
            "count": 20,  # Get more results to filter
            "language": "en",
            "format": "json",
        }

        response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Check if results exist
        if "results" not in data or not data["results"]:
            _geocoding_cache[cache_key] = None
            return None

        # Filter for US results
        us_results = [
            r for r in data["results"]
            if r.get("country_code") == "US"
        ]

        if not us_results:
            # No US results found
            _geocoding_cache[cache_key] = None
            return None

        # If state is specified, try to match it
        if state:
            for result in us_results:
                admin1 = result.get("admin1", "")
                # Match state abbreviation or full name
                if admin1 and _state_matches(admin1, state):
                    lat = round(result["latitude"], 4)
                    lon = round(result["longitude"], 4)
                    coords = (lat, lon)
                    _geocoding_cache[cache_key] = coords
                    return coords
            # State specified but no match - still return first US result
            # (user might have misspelled state)

        # Return first US result (usually largest city by population)
        result = us_results[0]
        lat = round(result["latitude"], 4)
        lon = round(result["longitude"], 4)
        coords = (lat, lon)
        _geocoding_cache[cache_key] = coords
        return coords

    except requests.RequestException:
        # Network error - don't cache, allow retry
        return None
    except (KeyError, ValueError, TypeError):
        # Data parsing error - cache as not found
        _geocoding_cache[cache_key] = None
        return None


# Common US state abbreviations to full names
STATE_ABBREVS = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}


def _state_matches(admin1: str, state: str) -> bool:
    """
    Check if the admin1 field from the API matches the given state.

    Handles both state abbreviations (NY) and full names (New York).
    """
    admin1_lower = admin1.lower()
    state_upper = state.upper()
    state_lower = state.lower()

    # Direct match (case insensitive)
    if admin1_lower == state_lower:
        return True

    # State is an abbreviation, check if admin1 is the full name
    if state_upper in STATE_ABBREVS:
        if admin1_lower == STATE_ABBREVS[state_upper].lower():
            return True

    # State is a full name, check if it matches admin1
    for abbrev, full_name in STATE_ABBREVS.items():
        if full_name.lower() == state_lower and admin1_lower == full_name.lower():
            return True
        # Or if user gave full name and admin1 matches
        if full_name.lower() == admin1_lower and state_upper == abbrev:
            return True

    return False


def get_coordinates_detailed(
    city_name: str,
    state: Optional[str] = None
) -> Optional[GeocodingResult]:
    """
    Get detailed geocoding result including name, state, population.

    Args:
        city_name: Name of the city
        state: Optional state abbreviation or name

    Returns:
        GeocodingResult with full details, or None if not found.
    """
    search_query = city_name
    if state:
        search_query = f"{city_name}, {state}"

    _rate_limit()

    try:
        params = {
            "name": search_query,
            "count": 10,
            "language": "en",
            "format": "json",
        }

        response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            return None

        # Prefer US results
        us_results = [
            r for r in data["results"]
            if r.get("country_code") == "US"
        ]

        result = us_results[0] if us_results else data["results"][0]

        return GeocodingResult(
            name=result.get("name", city_name),
            latitude=result["latitude"],
            longitude=result["longitude"],
            country=result.get("country", "Unknown"),
            admin1=result.get("admin1"),
            population=result.get("population"),
        )

    except (requests.RequestException, KeyError, ValueError, TypeError):
        return None


def clear_cache() -> None:
    """Clear the geocoding cache."""
    global _geocoding_cache
    _geocoding_cache = {}
