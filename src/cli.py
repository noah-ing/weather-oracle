"""Command-line interface for Weather Oracle.

Provides commands for weather prediction, model training, data collection,
and evaluation using Click and Rich for beautiful output.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional


console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="weather-oracle")
def cli():
    """Weather Oracle - Neural network weather forecasting.
    
    Use the commands below to predict weather, train models,
    collect data, or evaluate model performance.
    """
    pass


@cli.command()
@click.argument("city")
@click.option("--state", "-s", default=None, help="State abbreviation (e.g., NY, CA)")
@click.option("--days", "-d", default=1, type=int, help="Number of days to show (max 1 for now)")
def predict(city: str, state: Optional[str], days: int):
    """Get weather forecast for a city.
    
    Examples:
        weather-oracle predict "New York" --state NY
        weather-oracle predict Chicago -s IL
        weather-oracle predict "Los Angeles"
    """
    from src.inference.predictor import WeatherPredictor
    
    console.print(Panel.fit(
        f"[bold cyan]Weather Oracle Forecast[/bold cyan]\n"
        f"Location: {city}" + (f", {state}" if state else ""),
        border_style="cyan"
    ))
    
    try:
        with console.status("[bold green]Loading model and fetching weather data..."):
            predictor = WeatherPredictor()
            forecast = predictor.predict_city(city, state)
        
        if forecast is None:
            console.print(f"[red]Error: Could not find city '{city}'" + (f" in {state}" if state else "") + "[/red]")
            console.print("[yellow]Try adding a state abbreviation with --state[/yellow]")
            return
        
        # Print forecast summary
        console.print(f"\n[bold]Forecast for {forecast.location}[/bold]")
        console.print(f"Generated at: {forecast.generated_at.strftime('%Y-%m-%d %H:%M')}")
        console.print(f"Coordinates: ({forecast.latitude:.4f}, {forecast.longitude:.4f})\n")
        
        # Create hourly table
        table = Table(title="24-Hour Forecast", show_header=True, header_style="bold cyan")
        table.add_column("Time", style="dim")
        table.add_column("Temp (°F)", justify="right")
        table.add_column("Range", style="dim", justify="right")
        table.add_column("Precip %", justify="right")
        table.add_column("Wind (mph)", justify="right")
        table.add_column("Conditions")
        
        for h in forecast.hourly:
            # Color code temperature
            temp_color = "blue" if h.temperature < 50 else "green" if h.temperature < 75 else "red"
            
            # Color code precipitation
            precip_color = "red" if h.precip_probability > 50 else "yellow" if h.precip_probability > 20 else "green"
            
            table.add_row(
                h.timestamp.strftime("%H:%M"),
                f"[{temp_color}]{h.temperature:.0f}[/{temp_color}]",
                f"{h.temperature_low:.0f}-{h.temperature_high:.0f}",
                f"[{precip_color}]{h.precip_probability:.0f}%[/{precip_color}]",
                f"{h.wind_speed:.0f}",
                h.conditions,
            )
        
        console.print(table)
        
        # Print summary
        temps = [h.temperature for h in forecast.hourly]
        precips = [h.precip_probability for h in forecast.hourly]
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Temperature: {min(temps):.0f}°F - {max(temps):.0f}°F (avg {sum(temps)/len(temps):.0f}°F)")
        console.print(f"  Max precipitation chance: {max(precips):.0f}%")
        
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Train a model first: weather-oracle train[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option("--epochs", "-e", default=50, type=int, help="Number of training epochs")
@click.option("--lr", default=0.001, type=float, help="Learning rate")
@click.option("--patience", "-p", default=10, type=int, help="Early stopping patience")
@click.option("--batch-size", "-b", default=32, type=int, help="Training batch size")
def train(epochs: int, lr: float, patience: int, batch_size: int):
    """Train the weather prediction model.
    
    Examples:
        weather-oracle train --epochs 50
        weather-oracle train -e 100 --lr 0.0005
    """
    from src.training.trainer import train_model
    
    console.print(Panel.fit(
        f"[bold cyan]Weather Oracle Training[/bold cyan]\n"
        f"Epochs: {epochs} | LR: {lr} | Patience: {patience} | Batch: {batch_size}",
        border_style="cyan"
    ))
    
    train_model(epochs=epochs, lr=lr, patience=patience, batch_size=batch_size)


@cli.command()
@click.option("--days", "-d", default=730, type=int, help="Days of historical data to collect")
def collect(days: int):
    """Collect historical weather data.
    
    Downloads weather observations from Open-Meteo API for 20 US cities.
    Data is stored in the SQLite database for training.
    
    Examples:
        weather-oracle collect --days 365
        weather-oracle collect  # Default 2 years
    """
    from src.data.collector import collect_historical
    
    console.print(Panel.fit(
        f"[bold cyan]Weather Oracle Data Collection[/bold cyan]\n"
        f"Collecting {days} days of historical data",
        border_style="cyan"
    ))
    
    collect_historical(days_back=days)


@cli.command()
def evaluate():
    """Evaluate model performance.
    
    Runs full evaluation on test set, comparing to persistence baseline
    and computing metrics like MAE, RMSE, and precipitation accuracy.
    
    Results are saved to evaluation_results.json.
    """
    from src.evaluation.metrics import run_full_evaluation
    
    console.print(Panel.fit(
        "[bold cyan]Weather Oracle Evaluation[/bold cyan]",
        border_style="cyan"
    ))
    
    run_full_evaluation()


@cli.command()
@click.argument("question")
def ask(question: str):
    """Answer natural language weather questions.
    
    Examples:
        weather-oracle ask "Will it rain in Boston tomorrow?"
        weather-oracle ask "What's the forecast for Seattle?"
        weather-oracle ask "Is it going to be cold in Denver?"
    """
    from src.inference.predictor import WeatherPredictor
    from src.api.geocoding import get_coordinates
    import re
    
    console.print(Panel.fit(
        f"[bold cyan]Weather Oracle[/bold cyan]\n"
        f"Question: {question}",
        border_style="cyan"
    ))
    
    # Simple NLP to extract city and query type
    question_lower = question.lower()
    
    # Try to find a city in the question
    # Common US cities
    common_cities = [
        ("new york", "NY"), ("los angeles", "CA"), ("chicago", "IL"),
        ("houston", "TX"), ("phoenix", "AZ"), ("philadelphia", "PA"),
        ("san antonio", "TX"), ("san diego", "CA"), ("dallas", "TX"),
        ("san jose", "CA"), ("austin", "TX"), ("jacksonville", "FL"),
        ("san francisco", "CA"), ("columbus", "OH"), ("seattle", "WA"),
        ("denver", "CO"), ("boston", "MA"), ("miami", "FL"),
        ("atlanta", "GA"), ("portland", "OR"), ("las vegas", "NV"),
        ("detroit", "MI"), ("minneapolis", "MN"), ("charlotte", "NC"),
        ("kansas city", "MO"), ("salt lake city", "UT"), ("new orleans", "LA"),
    ]
    
    found_city = None
    found_state = None
    
    for city, state in common_cities:
        if city in question_lower:
            found_city = city.title()
            found_state = state
            break
    
    if not found_city:
        # Try to match "in <city>" pattern
        match = re.search(r"in\s+([A-Za-z\s]+?)(?:\s+tomorrow|\s+today|\?|$)", question)
        if match:
            found_city = match.group(1).strip().title()
    
    if not found_city:
        console.print("[yellow]I couldn't identify a city in your question.[/yellow]")
        console.print("Try: weather-oracle ask \"What's the weather in Chicago tomorrow?\"")
        return
    
    try:
        with console.status("[bold green]Analyzing weather data..."):
            predictor = WeatherPredictor()
            forecast = predictor.predict_city(found_city, found_state)
        
        if forecast is None:
            console.print(f"[yellow]Couldn't find weather data for {found_city}[/yellow]")
            return
        
        # Analyze the question type
        is_rain_question = any(w in question_lower for w in ["rain", "precipitation", "wet", "umbrella"])
        is_cold_question = any(w in question_lower for w in ["cold", "freezing", "chilly"])
        is_hot_question = any(w in question_lower for w in ["hot", "warm", "heat"])
        is_wind_question = any(w in question_lower for w in ["wind", "windy", "breezy"])
        
        # Get summary stats
        temps = [h.temperature for h in forecast.hourly]
        precips = [h.precip_probability for h in forecast.hourly]
        winds = [h.wind_speed for h in forecast.hourly]
        
        avg_temp = sum(temps) / len(temps)
        max_precip = max(precips)
        avg_wind = sum(winds) / len(winds)
        
        console.print(f"\n[bold]Weather for {forecast.location}[/bold]\n")
        
        # Generate natural response
        if is_rain_question:
            if max_precip > 70:
                console.print(f"[red]Yes, rain is likely![/red] Maximum precipitation chance is {max_precip:.0f}%.")
            elif max_precip > 40:
                console.print(f"[yellow]There's a moderate chance of rain.[/yellow] Maximum precipitation chance is {max_precip:.0f}%.")
            elif max_precip > 20:
                console.print(f"[green]There's a slight chance of rain.[/green] Maximum precipitation chance is {max_precip:.0f}%.")
            else:
                console.print(f"[green]It looks like it will be dry![/green] Maximum precipitation chance is only {max_precip:.0f}%.")
        
        elif is_cold_question:
            if min(temps) < 32:
                console.print(f"[cyan]Yes, it will be freezing![/cyan] Low of {min(temps):.0f}°F.")
            elif min(temps) < 50:
                console.print(f"[blue]Yes, it will be cold.[/blue] Temperatures between {min(temps):.0f}°F and {max(temps):.0f}°F.")
            else:
                console.print(f"[green]It won't be too cold.[/green] Temperatures between {min(temps):.0f}°F and {max(temps):.0f}°F.")
        
        elif is_hot_question:
            if max(temps) > 90:
                console.print(f"[red]Yes, it will be very hot![/red] High of {max(temps):.0f}°F.")
            elif max(temps) > 80:
                console.print(f"[yellow]It will be warm.[/yellow] High of {max(temps):.0f}°F.")
            else:
                console.print(f"[green]It won't be too hot.[/green] High of {max(temps):.0f}°F.")
        
        elif is_wind_question:
            if avg_wind > 20:
                console.print(f"[yellow]Yes, it will be quite windy![/yellow] Average wind speed {avg_wind:.0f} mph.")
            elif avg_wind > 10:
                console.print(f"[green]It will be breezy.[/green] Average wind speed {avg_wind:.0f} mph.")
            else:
                console.print(f"[green]Winds will be calm.[/green] Average wind speed {avg_wind:.0f} mph.")
        
        else:
            # General forecast
            console.print(f"Temperature: {min(temps):.0f}°F - {max(temps):.0f}°F (avg {avg_temp:.0f}°F)")
            console.print(f"Precipitation: {max_precip:.0f}% max chance")
            console.print(f"Wind: {avg_wind:.0f} mph average")
        
        # Show abbreviated hourly
        console.print("\n[dim]Next 6 hours:[/dim]")
        for h in forecast.hourly[:6]:
            console.print(f"  {h.timestamp.strftime('%H:%M')}: {h.temperature:.0f}°F, {h.precip_probability:.0f}% rain, {h.conditions}")
        
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Train a model first: weather-oracle train[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def status():
    """Show system status and data summary.
    
    Displays information about collected data, trained model,
    and system configuration.
    """
    from pathlib import Path
    from src.config import DATA_DIR, MODELS_DIR, DB_PATH
    from src.db.database import get_observation_count, init_db
    
    console.print(Panel.fit(
        "[bold cyan]Weather Oracle Status[/bold cyan]",
        border_style="cyan"
    ))
    
    # Database status
    init_db()
    obs_count = get_observation_count()
    db_exists = DB_PATH.exists()
    db_size = DB_PATH.stat().st_size / 1024 / 1024 if db_exists else 0
    
    # Model status
    model_path = MODELS_DIR / "best_model.pt"
    model_exists = model_path.exists()
    model_size = model_path.stat().st_size / 1024 / 1024 if model_exists else 0
    
    # Scaler status
    scaler_path = DATA_DIR / "scaler.pkl"
    scaler_exists = scaler_path.exists()
    
    # Print status table
    table = Table(title="System Status", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")
    
    # Database
    db_status = "[green]Ready[/green]" if db_exists and obs_count > 0 else "[yellow]Empty[/yellow]"
    table.add_row("Database", db_status, f"{obs_count:,} observations ({db_size:.1f} MB)")
    
    # Model
    model_status = "[green]Trained[/green]" if model_exists else "[red]Not trained[/red]"
    model_details = f"{model_size:.1f} MB" if model_exists else "Run: weather-oracle train"
    table.add_row("Model", model_status, model_details)
    
    # Scaler
    scaler_status = "[green]Ready[/green]" if scaler_exists else "[yellow]Not fitted[/yellow]"
    table.add_row("Scaler", scaler_status, "Required for inference")
    
    console.print(table)
    
    # Recommendations
    if not db_exists or obs_count < 10000:
        console.print("\n[yellow]Recommendation: Collect more data with 'weather-oracle collect'[/yellow]")
    if not model_exists:
        console.print("[yellow]Recommendation: Train a model with 'weather-oracle train'[/yellow]")


@cli.command(name="scan-markets")
@click.option("--max-series", "-m", default=100, type=int, help="Maximum series to query")
@click.option("--days", "-d", default=7, type=int, help="Days ahead to include")
def scan_markets(max_series: int, days: int):
    """Scan Kalshi for weather prediction markets.

    Shows all active weather markets from Kalshi that expire
    within the specified number of days.

    Examples:
        python -m src.cli scan-markets
        python -m src.cli scan-markets --days 3
    """
    from src.kalshi.scanner import scan_weather_markets

    console.print(Panel.fit(
        f"[bold cyan]Kalshi Weather Market Scanner[/bold cyan]\n"
        f"Scanning markets expiring in next {days} days",
        border_style="cyan"
    ))

    with console.status("[bold green]Fetching markets from Kalshi..."):
        markets = scan_weather_markets(max_series=max_series, days_ahead=days)

    if not markets:
        console.print("[yellow]No weather markets found[/yellow]")
        console.print("This could mean:")
        console.print("  - No markets expire in the specified time window")
        console.print("  - API rate limits prevented fetching data")
        return

    console.print(f"\n[green]Found {len(markets)} weather markets:[/green]\n")

    # Create table
    table = Table(title="Weather Markets", show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="cyan", max_width=25)
    table.add_column("Location")
    table.add_column("Date")
    table.add_column("Condition")
    table.add_column("Threshold", justify="right")
    table.add_column("Bid/Ask", justify="right")
    table.add_column("Volume", justify="right")

    for market in markets:
        # Color code based on condition
        cond_color = {
            "temp_high": "red",
            "temp_low": "blue",
            "rain": "cyan",
            "snow": "white",
        }.get(market.condition_type, "white")

        threshold_str = ""
        if market.threshold is not None:
            comparison_symbol = {"above": ">", "below": "<", "at_least": ">=", "at_most": "<="}.get(market.comparison or "", "")
            unit = "°F" if "temp" in market.condition_type else ""
            threshold_str = f"{comparison_symbol}{market.threshold}{unit}"

        table.add_row(
            market.ticker[:25],
            f"{market.location}" + (f", {market.state}" if market.state else ""),
            str(market.target_date) if market.target_date else "N/A",
            f"[{cond_color}]{market.condition_type}[/{cond_color}]",
            threshold_str,
            f"${market.yes_bid:.2f}/${market.yes_ask:.2f}",
            f"{market.volume:,}",
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(markets)} markets[/dim]")


@cli.command(name="find-edges")
@click.option("--min-edge", "-e", default=10, type=float, help="Minimum edge percentage")
@click.option("--max-series", "-m", default=100, type=int, help="Maximum series to query")
@click.option("--days", "-d", default=7, type=int, help="Days ahead to include")
def find_edges_cmd(min_edge: float, max_series: int, days: int):
    """Find edge opportunities in weather markets.

    Compares model predictions to Kalshi market prices to find
    opportunities where the model disagrees with the market.

    Examples:
        python -m src.cli find-edges
        python -m src.cli find-edges --min-edge 15
        python -m src.cli find-edges -e 5 --days 3
    """
    from src.kalshi.edge import find_edges

    console.print(Panel.fit(
        f"[bold cyan]Edge Calculator[/bold cyan]\n"
        f"Min edge: {min_edge}% | Days ahead: {days}",
        border_style="cyan"
    ))

    with console.status("[bold green]Scanning markets and calculating edges..."):
        edges = find_edges(min_edge=min_edge, max_series=max_series, days_ahead=days)

    if not edges:
        console.print("[yellow]No edge opportunities found above threshold[/yellow]")
        console.print("This could mean:")
        console.print("  - Markets are efficiently priced")
        console.print("  - No weather markets match our model's locations")
        console.print("  - API rate limits prevented fetching data")
        return

    console.print(f"\n[green]Found {len(edges)} edge opportunities:[/green]\n")

    # Create table
    table = Table(title="Edge Opportunities", show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="cyan", max_width=22)
    table.add_column("Location")
    table.add_column("Date")
    table.add_column("Kalshi", justify="right")
    table.add_column("Model", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("EV", justify="right")
    table.add_column("Side", justify="center")
    table.add_column("Conf")

    for edge in edges:
        # Color code edge - green for positive, red for negative
        edge_color = "green" if edge.edge_pct > 0 else "red"
        side_color = "green" if edge.side == "YES" else "red"
        conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(edge.confidence, "white")

        table.add_row(
            edge.market.ticker[:22],
            f"{edge.market.location}",
            str(edge.market.target_date) if edge.market.target_date else "N/A",
            f"{edge.kalshi_prob*100:.0f}%",
            f"{edge.model_prob*100:.0f}%",
            f"[{edge_color}]{edge.edge_pct:+.0f}%[/{edge_color}]",
            f"${edge.expected_value:.2f}",
            f"[{side_color}]{edge.side}[/{side_color}]",
            f"[{conf_color}]{edge.confidence}[/{conf_color}]",
        )

    console.print(table)

    # Show top opportunity details
    if edges:
        top = edges[0]
        console.print(f"\n[bold]Top Opportunity:[/bold]")
        console.print(f"  {top.explanation}")


@cli.command(name="alert-edges")
@click.option("--min-edge", "-e", default=15, type=float, help="Minimum edge percentage to alert")
@click.option("--max-series", "-m", default=100, type=int, help="Maximum series to query")
@click.option("--days", "-d", default=7, type=int, help="Days ahead to include")
def alert_edges_cmd(min_edge: float, max_series: int, days: int):
    """Find edges and send Telegram alerts.

    Finds edge opportunities and sends alerts to the configured
    Telegram chat for each opportunity.

    Examples:
        python -m src.cli alert-edges
        python -m src.cli alert-edges --min-edge 20
    """
    from src.kalshi.edge import find_edges
    from src.telegram.bot import send_edge_alert, send_edge_summary

    console.print(Panel.fit(
        f"[bold cyan]Edge Alert Sender[/bold cyan]\n"
        f"Min edge: {min_edge}% | Days ahead: {days}",
        border_style="cyan"
    ))

    with console.status("[bold green]Scanning markets and calculating edges..."):
        edges = find_edges(min_edge=min_edge, max_series=max_series, days_ahead=days)

    if not edges:
        console.print("[yellow]No edge opportunities found above threshold[/yellow]")
        return

    console.print(f"\n[green]Found {len(edges)} edge opportunities. Sending alerts...[/green]\n")

    # Send summary first
    summary_sent = send_edge_summary(edges)
    if summary_sent:
        console.print("[green]✓[/green] Summary sent to Telegram")
    else:
        console.print("[red]✗[/red] Failed to send summary (check Telegram config)")

    # Send individual alerts for top opportunities
    sent_count = 0
    failed_count = 0

    for edge in edges[:5]:  # Max 5 individual alerts
        success = send_edge_alert(edge)
        if success:
            sent_count += 1
            edge_color = "green" if edge.edge_pct > 0 else "red"
            console.print(f"[green]✓[/green] Alert sent: [{edge_color}]{edge.market.ticker}[/{edge_color}] ({edge.edge_pct:+.0f}% edge)")
        else:
            failed_count += 1
            console.print(f"[red]✗[/red] Failed: {edge.market.ticker}")

    console.print(f"\n[bold]Summary:[/bold] {sent_count} alerts sent, {failed_count} failed")

    if failed_count > 0 and sent_count == 0:
        console.print("[yellow]Check your Telegram configuration in .env file[/yellow]")


@cli.command(name="track-accuracy")
@click.option("--log", "-l", is_flag=True, help="Log new forecasts for today")
@click.option("--update", "-u", is_flag=True, help="Update with actual observations")
@click.option("--days", "-d", default=7, type=int, help="Rolling window for accuracy (days)")
def track_accuracy_cmd(log: bool, update: bool, days: int):
    """Track forecast accuracy across multiple sources.

    Logs predictions from GFS, ECMWF, NWS, etc. and calculates
    rolling accuracy metrics to identify the most reliable sources.

    Run hourly to log predictions, then after 24h the actuals are updated.

    Examples:
        python -m src.cli track-accuracy           # Show current accuracy
        python -m src.cli track-accuracy --log     # Log new forecasts
        python -m src.cli track-accuracy --update  # Update with actuals
        python -m src.cli track-accuracy -l -u     # Log + update + report
    """
    from src.tracking.forecast_tracker import (
        log_forecasts,
        update_actuals,
        get_accuracy_report,
        get_forecast_count,
        get_pending_actuals_count,
    )

    console.print(Panel.fit(
        "[bold cyan]Forecast Accuracy Tracker[/bold cyan]",
        border_style="cyan"
    ))

    # Log new forecasts if requested
    if log:
        console.print("\n[bold]Logging forecasts from all sources...[/bold]")
        with console.status("[green]Fetching forecasts..."):
            logged = log_forecasts()
        console.print(f"[green]Logged {logged} forecasts[/green]")

    # Update actuals if requested
    if update:
        console.print("\n[bold]Updating with actual observations...[/bold]")
        with console.status("[green]Fetching historical data..."):
            updated = update_actuals()
        console.print(f"[green]Updated {updated} forecasts with actuals[/green]")

    # Always show the accuracy report
    console.print(f"\n[bold]Accuracy Report ({days}-day rolling window):[/bold]")
    report = get_accuracy_report(days=days)
    console.print(report)

    # Show status
    total = get_forecast_count()
    pending = get_pending_actuals_count()

    console.print(f"\n[dim]Total forecasts logged: {total:,}[/dim]")
    console.print(f"[dim]Pending actuals: {pending:,}[/dim]")


@cli.command(name="watch")
@click.option("--interval", "-i", default=60, type=int, help="Minutes between scans")
@click.option("--min-edge", "-e", default=10, type=float, help="Minimum edge percentage to alert")
@click.option("--max-series", "-m", default=100, type=int, help="Maximum series to query")
@click.option("--days", "-d", default=7, type=int, help="Days ahead to include")
def watch_cmd(interval: int, min_edge: float, max_series: int, days: int):
    """Run continuous edge scanner with Telegram alerts.

    Continuously scans Kalshi weather markets for edge opportunities
    and sends Telegram alerts when new opportunities are found.
    Tracks already-alerted markets to avoid spam.

    Press Ctrl+C to stop gracefully.

    Examples:
        python -m src.cli watch
        python -m src.cli watch --interval 30 --min-edge 15
        python -m src.cli watch -i 1 -e 5  # Quick test mode
    """
    from src.kalshi.scheduler import run_scanner

    console.print(Panel.fit(
        f"[bold cyan]Weather Oracle Edge Scanner[/bold cyan]\n"
        f"Interval: {interval} min | Min edge: {min_edge}% | Days: {days}",
        border_style="cyan"
    ))

    run_scanner(
        interval_minutes=interval,
        min_edge=min_edge,
        max_series=max_series,
        days_ahead=days,
    )


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
