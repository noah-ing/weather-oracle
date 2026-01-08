"""Regional city groupings for V3 regional model training.

Defines 5 US regions based on climate patterns and geography.
Each region has 3-5 cities with similar weather characteristics.
"""

from typing import Dict, List, Optional


# Regional city groupings - each region shares similar climate patterns
REGIONS: Dict[str, List[tuple[str, str]]] = {
    "northeast": [
        ("New York", "NY"),
        ("Boston", "MA"),
        ("Philadelphia", "PA"),
    ],
    "southeast": [
        ("Miami", "FL"),
        ("Atlanta", "GA"),
        ("Charlotte", "NC"),
        ("New Orleans", "LA"),
    ],
    "midwest": [
        ("Chicago", "IL"),
        ("Detroit", "MI"),
        ("Minneapolis", "MN"),
        ("Kansas City", "MO"),
    ],
    "southwest": [
        ("Phoenix", "AZ"),
        ("Las Vegas", "NV"),
        ("Denver", "CO"),
        ("Dallas", "TX"),
        ("Houston", "TX"),
    ],
    "west": [
        ("Los Angeles", "CA"),
        ("San Francisco", "CA"),
        ("Seattle", "WA"),
        ("Portland", "OR"),
        ("Salt Lake City", "UT"),
    ],
}


# Flattened lookup: city -> region
_CITY_TO_REGION: Dict[str, str] = {}
for region_name, cities in REGIONS.items():
    for city, state in cities:
        _CITY_TO_REGION[city.lower()] = region_name
        _CITY_TO_REGION[f"{city}, {state}".lower()] = region_name


def get_region(city: str) -> Optional[str]:
    """
    Get the region name for a given city.

    Args:
        city: City name (e.g., "Chicago" or "Chicago, IL")

    Returns:
        Region name (e.g., "midwest") or None if city not found
    """
    return _CITY_TO_REGION.get(city.lower())


def get_cities_in_region(region: str) -> List[tuple[str, str]]:
    """
    Get all cities in a given region.

    Args:
        region: Region name (e.g., "midwest")

    Returns:
        List of (city, state) tuples for that region
    """
    return REGIONS.get(region.lower(), [])


def get_all_regions() -> List[str]:
    """
    Get list of all region names.

    Returns:
        List of region names
    """
    return list(REGIONS.keys())


def get_city_names_in_region(region: str) -> List[str]:
    """
    Get city names (without state) in a given region.

    Args:
        region: Region name (e.g., "midwest")

    Returns:
        List of city names
    """
    return [city for city, _ in get_cities_in_region(region)]


def get_formatted_cities_in_region(region: str) -> List[str]:
    """
    Get formatted "City, State" strings for a region.

    Args:
        region: Region name (e.g., "midwest")

    Returns:
        List of "City, State" strings
    """
    return [f"{city}, {state}" for city, state in get_cities_in_region(region)]
