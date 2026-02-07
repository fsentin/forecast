"""
Utility functions for data validation, preprocessing, and visualization.

This module provides helper functions for:
- Input validation (column types, time series size)
- Time series preprocessing (gap filling, outlier detection)
- Model evaluation (metrics calculation)
- Plotting and visualization
"""

from .input_validation import (
    validate_date_column,
    validate_numeric_column,
    validate_columns,
    validate_timeseries_size,
)

from .timeseries import (
    is_equally_spaced,
    fill_gaps_interpolate,
    fill_gaps_zero
)

from .model_evaluation import (
    calculate_metrics
)

from .plotting import (
    plot_forecast,
    plot_model_comparison
)

__all__ = [
    # Input validation
    "validate_date_column",
    "validate_numeric_column",
    "validate_columns",
    "validate_timeseries_size",
    # Time series utilities
    "is_equally_spaced",
    "fill_gaps_interpolate",
    "fill_gaps_zero",
    # Model evaluation
    "calculate_metrics",
    # Plotting
    "plot_forecast",
    "plot_model_comparison",
]