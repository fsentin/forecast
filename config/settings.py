"""Application configuration and constants."""

### Random Seed ###
import plotly

RANDOM_SEED = 22

### Chart Color Configuration ###
CHART_COLORS = {
    'historical': '#3865fa',
    'forecast': '#FF6B6B',
    "test": "#2ca02c",
    "prediction": "#ff7f0e",
    "split_line": "gray",
}

MODEL_COMPARISON_PALETTE = plotly.colors.qualitative.Plotly[1:]


### Training Defaults ###
TRAIN_PCT_DEFAULT = 80
TRAIN_PCT_MIN = 70
TRAIN_PCT_MAX = 90

### Forecast Defaults ###
HORIZON_DEFAULT = 30
HORIZON_MIN = 1
HORIZON_MAX = 365

### Upload Data Limitiations ###
MIN_DATA_POINTS = 20
MAX_DATA_POINTS = 50_000

### Page Configuration ###
PAGE_CONFIG = {
    "page_title": "Forecast",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}