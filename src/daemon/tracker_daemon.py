"""Automated accuracy tracking daemon.

This daemon runs continuously to:
- Log forecasts from all sources every 6 hours
- Update with actual observations daily
- Calculate rolling accuracy metrics
- Recalculate bias corrections weekly
- Send weekly accuracy reports to Telegram

The daemon can be run directly or managed as a system service.

Usage:
    Direct:
        python -m src.daemon.tracker_daemon

    Via CLI:
        python -m src.cli start-tracker

    As service:
        launchctl load ~/Library/LaunchAgents/com.weather-oracle.tracker.plist
"""

import signal
import sys
import time
import threading
import schedule
from datetime import datetime, timedelta
from typing import Optional

from src.config import DATA_DIR
from src.tracking.forecast_tracker import (
    log_forecasts,
    update_actuals,
    get_accuracy_report,
    get_forecast_count,
)
from src.calibration.bias_correction import (
    update_all_bias_corrections,
)
from src.telegram.bot import send_alert


# Daemon state
_stop_event = threading.Event()
_daemon_thread: Optional[threading.Thread] = None


class TrackerDaemon:
    """Background daemon for automated forecast tracking.

    Schedules and runs:
    - Forecast logging every 6 hours (00:00, 06:00, 12:00, 18:00)
    - Actuals update daily at 23:00
    - Bias recalculation weekly on Sunday at 01:00
    - Weekly accuracy report on Sunday at 02:00

    Attributes:
        forecast_interval_hours: Hours between forecast logs (default 6)
        running: Whether the daemon is currently running
        last_forecast_log: Time of last forecast log
        last_actuals_update: Time of last actuals update
        last_bias_update: Time of last bias recalculation
        last_weekly_report: Time of last weekly report
    """

    def __init__(self, forecast_interval_hours: int = 6):
        """Initialize the tracker daemon.

        Args:
            forecast_interval_hours: Hours between forecast logging runs
        """
        self.forecast_interval_hours = forecast_interval_hours
        self.running = False
        self.last_forecast_log: Optional[datetime] = None
        self.last_actuals_update: Optional[datetime] = None
        self.last_bias_update: Optional[datetime] = None
        self.last_weekly_report: Optional[datetime] = None

        # Stats
        self.total_forecasts_logged = 0
        self.total_actuals_updated = 0
        self.started_at: Optional[datetime] = None

    def _log_forecasts(self) -> None:
        """Log forecasts from all sources."""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Logging forecasts...")

        try:
            count = log_forecasts()
            self.total_forecasts_logged += count
            self.last_forecast_log = datetime.now()
            print(f"  Logged {count} forecasts (total: {self.total_forecasts_logged})")
        except Exception as e:
            print(f"  Error logging forecasts: {e}")

    def _update_actuals(self) -> None:
        """Update forecasts with actual observations."""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating actuals...")

        try:
            count = update_actuals()
            self.total_actuals_updated += count
            self.last_actuals_update = datetime.now()
            print(f"  Updated {count} forecasts with actuals (total: {self.total_actuals_updated})")
        except Exception as e:
            print(f"  Error updating actuals: {e}")

    def _recalculate_bias(self) -> None:
        """Recalculate bias corrections for all sources/locations."""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recalculating bias corrections...")

        try:
            update_all_bias_corrections()
            self.last_bias_update = datetime.now()
            print("  Bias corrections updated")
        except Exception as e:
            print(f"  Error recalculating bias: {e}")

    def _send_weekly_report(self) -> None:
        """Send weekly accuracy report to Telegram."""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sending weekly report...")

        try:
            report = get_accuracy_report(days=7)
            total = get_forecast_count()

            message = (
                "📊 <b>Weekly Accuracy Report</b>\n\n"
                f"{report}\n\n"
                f"Total forecasts logged: {total:,}\n"
                f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            success = send_alert(message)
            self.last_weekly_report = datetime.now()

            if success:
                print("  Weekly report sent to Telegram")
            else:
                print("  Failed to send weekly report (check Telegram config)")

        except Exception as e:
            print(f"  Error sending weekly report: {e}")

    def _setup_schedule(self) -> None:
        """Set up the scheduled tasks."""
        # Clear any existing jobs
        schedule.clear()

        # Forecast logging every 6 hours at fixed times
        schedule.every().day.at("00:00").do(self._log_forecasts)
        schedule.every().day.at("06:00").do(self._log_forecasts)
        schedule.every().day.at("12:00").do(self._log_forecasts)
        schedule.every().day.at("18:00").do(self._log_forecasts)

        # Actuals update daily at 23:00 (gives time for all observations to arrive)
        schedule.every().day.at("23:00").do(self._update_actuals)

        # Bias recalculation weekly on Sunday at 01:00
        schedule.every().sunday.at("01:00").do(self._recalculate_bias)

        # Weekly report on Sunday at 02:00
        schedule.every().sunday.at("02:00").do(self._send_weekly_report)

        print("Schedule configured:")
        print("  - Forecast logging: 00:00, 06:00, 12:00, 18:00")
        print("  - Actuals update: daily at 23:00")
        print("  - Bias recalculation: Sunday at 01:00")
        print("  - Weekly report: Sunday at 02:00")

    def start(self, run_immediately: bool = True) -> None:
        """Start the tracker daemon.

        Args:
            run_immediately: Whether to run forecast logging immediately on start
        """
        print("=" * 60)
        print("Weather Oracle Tracker Daemon")
        print("=" * 60)

        self.started_at = datetime.now()
        self.running = True

        # Set up schedule
        self._setup_schedule()

        # Run immediately if requested
        if run_immediately:
            print("\nRunning initial forecast log...")
            self._log_forecasts()

        print(f"\nDaemon started at {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to stop.\n")

        # Main loop
        try:
            while self.running and not _stop_event.is_set():
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\nReceived shutdown signal...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the tracker daemon."""
        self.running = False
        schedule.clear()

        # Print summary
        if self.started_at:
            uptime = datetime.now() - self.started_at
            print("\n" + "=" * 60)
            print("Tracker Daemon Stopped")
            print("=" * 60)
            print(f"Uptime: {uptime}")
            print(f"Forecasts logged: {self.total_forecasts_logged}")
            print(f"Actuals updated: {self.total_actuals_updated}")
            print("=" * 60)

    def status(self) -> dict:
        """Get current daemon status.

        Returns:
            Dict with daemon status information
        """
        uptime = None
        if self.started_at:
            uptime = datetime.now() - self.started_at

        return {
            "running": self.running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": uptime.total_seconds() if uptime else None,
            "total_forecasts_logged": self.total_forecasts_logged,
            "total_actuals_updated": self.total_actuals_updated,
            "last_forecast_log": self.last_forecast_log.isoformat() if self.last_forecast_log else None,
            "last_actuals_update": self.last_actuals_update.isoformat() if self.last_actuals_update else None,
            "last_bias_update": self.last_bias_update.isoformat() if self.last_bias_update else None,
            "last_weekly_report": self.last_weekly_report.isoformat() if self.last_weekly_report else None,
        }


def _handle_signal(signum, frame):
    """Handle shutdown signals."""
    print(f"\nReceived signal {signum}, stopping daemon...")
    _stop_event.set()


def run_tracker_daemon(
    run_immediately: bool = True,
    forecast_interval_hours: int = 6,
) -> None:
    """Run the tracker daemon in the current process.

    This is the main entry point for running the daemon.

    Args:
        run_immediately: Whether to run forecast logging immediately
        forecast_interval_hours: Hours between forecast logs
    """
    # Register signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    daemon = TrackerDaemon(forecast_interval_hours=forecast_interval_hours)
    daemon.start(run_immediately=run_immediately)


def start_daemon_thread() -> bool:
    """Start the tracker daemon in a background thread.

    Returns:
        True if started successfully, False if already running
    """
    global _daemon_thread

    if _daemon_thread is not None and _daemon_thread.is_alive():
        return False

    _stop_event.clear()
    _daemon_thread = threading.Thread(target=run_tracker_daemon, daemon=True)
    _daemon_thread.start()

    return True


def stop_daemon_thread() -> bool:
    """Stop the tracker daemon thread.

    Returns:
        True if stopped successfully, False if not running
    """
    global _daemon_thread

    if _daemon_thread is None or not _daemon_thread.is_alive():
        return False

    _stop_event.set()
    _daemon_thread.join(timeout=5)

    return True


def is_daemon_running() -> bool:
    """Check if the daemon thread is running.

    Returns:
        True if daemon is running
    """
    return _daemon_thread is not None and _daemon_thread.is_alive()


# Launchd plist template
LAUNCHD_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.weather-oracle.tracker</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>src.cli</string>
        <string>start-tracker</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{project_dir}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{log_dir}/tracker.log</string>

    <key>StandardErrorPath</key>
    <string>{log_dir}/tracker.error.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
"""


def generate_launchd_plist(
    project_dir: str = None,
    python_path: str = None,
    log_dir: str = None,
) -> str:
    """Generate a launchd plist file for auto-starting the daemon.

    Args:
        project_dir: Path to the project directory (default: current working dir)
        python_path: Path to Python interpreter (default: sys.executable)
        log_dir: Path for log files (default: project_dir/logs)

    Returns:
        Path to the generated plist file
    """
    import os

    if project_dir is None:
        project_dir = os.getcwd()

    if python_path is None:
        python_path = sys.executable

    if log_dir is None:
        log_dir = os.path.join(project_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

    plist_content = LAUNCHD_PLIST.format(
        python_path=python_path,
        project_dir=project_dir,
        log_dir=log_dir,
    )

    # Write to LaunchAgents
    plist_path = os.path.expanduser(
        "~/Library/LaunchAgents/com.weather-oracle.tracker.plist"
    )

    # Create LaunchAgents directory if needed
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)

    with open(plist_path, "w") as f:
        f.write(plist_content)

    print(f"Generated launchd plist: {plist_path}")
    print("\nTo install the service:")
    print(f"  launchctl load {plist_path}")
    print("\nTo uninstall:")
    print(f"  launchctl unload {plist_path}")
    print("\nTo check status:")
    print("  launchctl list | grep weather-oracle")

    return plist_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather Oracle Tracker Daemon")
    parser.add_argument(
        "--no-immediate",
        action="store_true",
        help="Don't run forecast log immediately on start",
    )
    parser.add_argument(
        "--generate-plist",
        action="store_true",
        help="Generate launchd plist file for auto-start",
    )

    args = parser.parse_args()

    if args.generate_plist:
        generate_launchd_plist()
    else:
        run_tracker_daemon(run_immediately=not args.no_immediate)
