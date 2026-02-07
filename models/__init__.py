from .arima import ARIMAForecaster
from .base import ForecastModel
from .nbeats import NBEATSForecaster
from .prophet import ProphetForecaster

# Model registry
AVAILABLE_MODELS = {
    'ARIMA': ARIMAForecaster,
    'PROPHET': ProphetForecaster,
    'NBEATS': NBEATSForecaster
}

__all__ = ['ForecastModel', 'ARIMAForecaster', 'ProphetForecaster', 'NBEATSForecaster', 'AVAILABLE_MODELS']