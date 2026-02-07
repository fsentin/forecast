"""
Forecast - Time Series Forecasting Application
================================================

A Streamlit-based application for time series forecasting using multiple models:
- ARIMA (AutoRegressive Integrated Moving Average)
- Prophet (Facebook's forecasting tool)
- N-BEATS (Neural Basis Expansion Analysis for Time Series)

Main Components
---------------
- **models**: Forecasting model implementations
- **services**: Business logic and data processing
- **ui**: Streamlit user interface components
- **utils**: Utility functions for validation, preprocessing, and plotting
- **config**: Application configuration and constants
- **splitters**: Train/test split strategies
- **state**: Application state management

Usage
-----
Run the application with:
    streamlit run app.py

Author: Fani
"""

__version__ = "1.0.0"
__author__ = "Fani"

# Import main components for easier access
from config.settings import (
    MIN_DATA_POINTS,
    MAX_DATA_POINTS,
    HORIZON_DEFAULT,
    TRAIN_PCT_DEFAULT
)

__all__ = [
    "MIN_DATA_POINTS",
    "MAX_DATA_POINTS", 
    "HORIZON_DEFAULT",
    "TRAIN_PCT_DEFAULT",
]