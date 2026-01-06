"""Historical weather data collector script.

Collects 2 years of historical weather data for 20 diverse US cities
and stores it in the SQLite database.
"""

import time
from datetime import date, timedelta
from typing import Optional

from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from src.api.open_meteo import get_historical, HistoricalData
from src.api.geocoding import get_coordinates
from src.db.database import init_db, insert_observations_batch, get_observation_count


console = Console()

# 20 diverse US cities covering different climates and regions
TRAINING_CITIES = [
    # Northeast
    ("New York", "NY"),
    ("Boston", "MA"),
    # Southeast
    ("Miami", "FL"),
    ("Atlanta", "GA"),
    ("Charlotte", "NC"),
    # Midwest
    ("Chicago", "IL"),
    ("Detroit", "MI"),
    ("Minneapolis", "MN"),
    # Southwest
    ("Phoenix", "AZ"),
    ("Las Vegas", "NV"),
    ("Denver", "CO"),
    # West Coast
    ("Los Angeles", "CA"),
    ("San Francisco", "CA"),
    ("Seattle", "WA"),
    ("Portland", "OR"),
    # South/Gulf
    ("Houston", "TX"),
    ("Dallas", "TX"),
    ("New Orleans", "LA"),
    # Mountain/Plains
    ("Salt Lake City", "UT"),
    ("Kansas City", "MO"),
]


def collect_city_data(
    city: str,
    state: str,
    start_date: date,
    end_date: date,
    max_retries: int = 3,
) -> int:
    """
    Collect historical data for a single city.

    Args:
        city: City name
        state: State abbreviation
        start_date: Start date for data collection
        end_date: End date for data collection
        max_retries: Maximum retry attempts on failure

    Returns:
        Number of observations inserted
    """
    # Get city coordinates
    coords = get_coordinates(city, state)
    if coords is None:
        console.print(f"[yellow]Warning: Could not find coordinates for {city}, {state}[/yellow]")
        return 0

    lat, lon = coords
    city_name = f"{city}, {state}"

    # Open-Meteo historical API can handle up to ~1 year per request efficiently
    # Split into chunks of 90 days for reliability
    chunk_days = 90
    total_inserted = 0
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)

        for attempt in range(max_retries):
            try:
                historical = get_historical(lat, lon, current_start, current_end)

                # Convert to observation records
                observations = []
                for i, timestamp in enumerate(historical.timestamps):
                    observations.append({
                        "city": city_name,
                        "lat": lat,
                        "lon": lon,
                        "timestamp": timestamp,
                        "temp": historical.temperatures[i],
                        "humidity": historical.humidity[i],
                        "wind_speed": historical.wind_speed[i],
                        "precipitation": historical.precipitation[i],
                        "pressure": historical.pressure[i],
                        "cloud_cover": historical.cloud_cover[i],
                    })

                # Batch insert
                inserted = insert_observations_batch(observations)
                total_inserted += len(observations)
                break  # Success, exit retry loop

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff
                    console.print(f"[yellow]Retry {attempt + 1}/{max_retries} for {city_name}: {e}[/yellow]")
                    time.sleep(wait_time)
                else:
                    console.print(f"[red]Failed to collect data for {city_name} ({current_start} to {current_end}): {e}[/red]")

        current_start = current_end + timedelta(days=1)

    return total_inserted


def collect_historical(days_back: int = 730) -> int:
    """
    Collect historical weather data for all training cities.

    Args:
        days_back: Number of days of historical data to collect (default 730 = 2 years)

    Returns:
        Total number of observations inserted
    """
    # Initialize the database
    init_db()

    end_date = date.today() - timedelta(days=1)  # Yesterday (latest available)
    start_date = end_date - timedelta(days=days_back)

    console.print(f"\n[bold cyan]Weather Oracle - Historical Data Collector[/bold cyan]")
    console.print(f"Collecting data from {start_date} to {end_date}")
    console.print(f"Cities: {len(TRAINING_CITIES)}")
    console.print(f"Expected observations: ~{len(TRAINING_CITIES) * days_back * 24:,} hourly records\n")

    total_inserted = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            "[cyan]Collecting data...",
            total=len(TRAINING_CITIES),
        )

        for city, state in TRAINING_CITIES:
            progress.update(main_task, description=f"[cyan]Collecting {city}, {state}...")

            inserted = collect_city_data(city, state, start_date, end_date)
            total_inserted += inserted

            progress.advance(main_task)

    # Final stats
    final_count = get_observation_count()
    console.print(f"\n[bold green]Collection complete![/bold green]")
    console.print(f"New observations inserted: {total_inserted:,}")
    console.print(f"Total observations in database: {final_count:,}")

    return total_inserted


def main():
    """Main entry point for the collector script."""
    import sys

    # Default to 2 years, but allow override via command line
    days_back = 730
    if len(sys.argv) > 1:
        try:
            days_back = int(sys.argv[1])
        except ValueError:
            console.print("[red]Usage: python -m src.data.collector [days_back][/red]")
            sys.exit(1)

    collect_historical(days_back=days_back)


if __name__ == "__main__":
    main()
