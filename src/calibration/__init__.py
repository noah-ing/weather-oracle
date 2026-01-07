"""Calibration module for bias correction and probabilistic forecasting."""

from src.calibration.bias_correction import (
    BiasCorrection,
    init_bias_corrections_table,
    calculate_bias,
    update_all_bias_corrections,
    get_bias_correction,
    correct_forecast,
    get_all_bias_corrections,
)

from src.calibration.probability import (
    ErrorDistribution,
    CalibrationConfig,
    get_error_distribution,
    get_adjusted_std,
    get_threshold_probability,
    get_confidence_interval,
    get_distribution_summary,
    get_calibration_report,
    calibrate_market_probability,
)

__all__ = [
    # Bias correction
    "BiasCorrection",
    "init_bias_corrections_table",
    "calculate_bias",
    "update_all_bias_corrections",
    "get_bias_correction",
    "correct_forecast",
    "get_all_bias_corrections",
    # Probabilistic calibration
    "ErrorDistribution",
    "CalibrationConfig",
    "get_error_distribution",
    "get_adjusted_std",
    "get_threshold_probability",
    "get_confidence_interval",
    "get_distribution_summary",
    "get_calibration_report",
    "calibrate_market_probability",
]
