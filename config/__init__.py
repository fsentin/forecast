"""
Configuration module for the forecast application.

This module contains application-wide settings, constants, and configuration values.
"""

from .settings import (
    RANDOM_SEED,
    CHART_COLORS,
    MODEL_COMPARISON_PALETTE,
    TRAIN_PCT_DEFAULT,
    TRAIN_PCT_MIN,
    TRAIN_PCT_MAX,
    HORIZON_DEFAULT,
    HORIZON_MIN,
    HORIZON_MAX,
    MIN_DATA_POINTS,
    MAX_DATA_POINTS,
    PAGE_CONFIG
)

__all__ = [
    "RANDOM_SEED",
    "CHART_COLORS",
    "MODEL_COMPARISON_PALETTE",
    "TRAIN_PCT_DEFAULT",
    "TRAIN_PCT_MIN",
    "TRAIN_PCT_MAX",
    "HORIZON_DEFAULT",
    "HORIZON_MIN",
    "HORIZON_MAX",
    "MIN_DATA_POINTS",
    "MAX_DATA_POINTS",
    "PAGE_CONFIG"
]