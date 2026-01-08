"""Historical weather data collector script.

Collects up to 5 years of historical weather data for 20 diverse US cities
and stores it in the SQLite database with extended atmospheric features.
"""

import time
from datetime import date, timedelta, datetime
from typing import Optional

from rich.console import Console
from rich.progress import (
    Progress,
    TaskID,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table

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
    progress: Optional[Progress] = None,
    chunk_task: Optional[TaskID] = None,
) -> int:
    """
    Collect historical data for a single city.

    Args:
        city: City name
        state: State abbreviation
        start_date: Start date for data collection
        end_date: End date for data collection
        max_retries: Maximum retry attempts on failure
        progress: Rich Progress instance for chunk updates
        chunk_task: Task ID for chunk progress

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

    # Calculate total chunks for this city
    total_days = (end_date - start_date).days
    total_chunks = (total_days + chunk_days - 1) // chunk_days
    chunk_num = 0

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
        chunk_num += 1

        for attempt in range(max_retries):
            try:
                historical = get_historical(lat, lon, current_start, current_end)

                # Convert to observation records with V3 extended features
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
                        "wind_direction": historical.wind_direction[i],
                        "precipitation": historical.precipitation[i],
                        "pressure_msl": historical.pressure_msl[i],
                        "surface_pressure": historical.surface_pressure[i],
                        "cloud_cover": historical.cloud_cover[i],
                        "cloud_cover_low": historical.cloud_cover_low[i],
                        "cloud_cover_mid": historical.cloud_cover_mid[i],
                        "cloud_cover_high": historical.cloud_cover_high[i],
                        "dewpoint": historical.dewpoint[i],
                    })

                # Batch insert
                inserted = insert_observations_batch(observations)
                total_inserted += len(observations)

                # Update chunk progress if available
                if progress and chunk_task is not None:
                    progress.update(chunk_task, advance=1)

                break  # Success, exit retry loop

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff
                    console.print(f"[yellow]Retry {attempt + 1}/{max_retries} for {city_name}: {e}[/yellow]")
                    time.sleep(wait_time)
                else:
                    console.print(f"[red]Failed to collect data for {city_name} ({current_start} to {current_end}): {e}[/red]")
                    # Still advance progress to avoid stuck progress bar
                    if progress and chunk_task is not None:
                        progress.update(chunk_task, advance=1)

        current_start = current_end + timedelta(days=1)

    return total_inserted


def collect_historical(days_back: int = 1825) -> int:
    """
    Collect historical weather data for all training cities.

    Args:
        days_back: Number of days of historical data to collect (default 1825 = 5 years)

    Returns:
        Total number of observations inserted
    """
    # Initialize the database
    init_db()

    end_date = date.today() - timedelta(days=1)  # Yesterday (latest available)
    start_date = end_date - timedelta(days=days_back)

    # Calculate total chunks for progress tracking
    chunk_days = 90
    chunks_per_city = (days_back + chunk_days - 1) // chunk_days
    total_chunks = len(TRAINING_CITIES) * chunks_per_city

    # Estimate time (roughly 2-3 seconds per chunk due to rate limiting)
    estimated_minutes = (total_chunks * 2.5) / 60

    console.print(f"\n[bold cyan]Weather Oracle V3 - Extended Data Collector[/bold cyan]")
    console.print(f"Collecting data from {start_date} to {end_date}")
    console.print(f"Cities: {len(TRAINING_CITIES)}")
    console.print(f"Days: {days_back} ({days_back / 365:.1f} years)")
    console.print(f"Expected observations: ~{len(TRAINING_CITIES) * days_back * 24:,} hourly records")
    console.print(f"Estimated time: ~{estimated_minutes:.0f} minutes")
    console.print(f"Features: temp, humidity, wind_speed, wind_direction, precip,")
    console.print(f"          pressure_msl, surface_pressure, cloud_cover (low/mid/high), dewpoint\n")

    total_inserted = 0
    start_time = datetime.now()
    cities_completed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        # Main city-level progress
        city_task = progress.add_task(
            "[cyan]Cities",
            total=len(TRAINING_CITIES),
        )

        # Chunk-level progress (more granular)
        chunk_task = progress.add_task(
            "[blue]Chunks",
            total=total_chunks,
        )

        for city, state in TRAINING_CITIES:
            progress.update(city_task, description=f"[cyan]Collecting {city}, {state}...")

            inserted = collect_city_data(
                city, state, start_date, end_date,
                progress=progress, chunk_task=chunk_task
            )
            total_inserted += inserted
            cities_completed += 1

            progress.update(city_task, advance=1)

            # Show intermediate stats every 5 cities
            if cities_completed % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = total_inserted / max(elapsed, 0.1)
                console.print(
                    f"  [dim]Progress: {cities_completed}/{len(TRAINING_CITIES)} cities, "
                    f"{total_inserted:,} records, {elapsed:.1f}min elapsed, "
                    f"{rate:.0f} records/min[/dim]"
                )

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    final_count = get_observation_count()

    console.print(f"\n[bold green]Collection complete![/bold green]")
    console.print(f"Time elapsed: {elapsed:.1f} minutes")
    console.print(f"New observations inserted: {total_inserted:,}")
    console.print(f"Total observations in database: {final_count:,}")
    console.print(f"Collection rate: {total_inserted / max(elapsed, 0.1):.0f} records/min")

    return total_inserted


def main():
    """Main entry point for the collector script."""
    import sys

    # Default to 5 years (1825 days), but allow override via command line
    days_back = 1825
    if len(sys.argv) > 1:
        try:
            days_back = int(sys.argv[1])
        except ValueError:
            console.print("[red]Usage: python -m src.data.collector [days_back][/red]")
            console.print("[dim]Default: 1825 (5 years)[/dim]")
            sys.exit(1)

    collect_historical(days_back=days_back)


if __name__ == "__main__":
    main()
