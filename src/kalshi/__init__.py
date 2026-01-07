"""Kalshi prediction market integration modules."""

from src.kalshi.scanner import WeatherMarket, scan_weather_markets
from src.kalshi.edge import EdgeOpportunity, calculate_edge, find_edges
from src.kalshi.edge_v2 import CalibratedEdge, calculate_calibrated_edge, find_calibrated_edges
from src.kalshi.scheduler import run_scanner, run_single_scan

__all__ = [
    "WeatherMarket",
    "scan_weather_markets",
    # V1 edge (legacy)
    "EdgeOpportunity",
    "calculate_edge",
    "find_edges",
    # V2 calibrated edge
    "CalibratedEdge",
    "calculate_calibrated_edge",
    "find_calibrated_edges",
    # Scheduler
    "run_scanner",
    "run_single_scan",
]
