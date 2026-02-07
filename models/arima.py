# models/arima.py
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np
import pandas as pd
from .base import ForecastModel
from splitters import TimeSeriesSplitter

class ARIMAForecaster(ForecastModel):
    """ARIMA (AutoRegressive Integrated Moving Average) forecasting model.
    
    ARIMA combines three components:
    - AR(p): Autoregression - uses past values
    - I(d): Integration - differences the data to make it stationary
    - MA(q): Moving Average - uses past forecast errors
    
    Uses statsmodels ARIMA and provides parameter suggestions based on
    statistical tests (ADF, ACF, PACF).
    
    Attributes:
        order: Tuple of (p, d, q) for statsmodels.
        model: Fitted statsmodels ARIMA model instance (None until fitted).
        _last_date: Last date in training data (used for forecasting).
        _freq: Inferred frequency of time series (e.g., 'D', 'M', 'H').
    """
    def __init__(self, p=1, d=1, q=1) -> None:
        """Initialize ARIMA model with (p, d, q) order.
        
        Args:
            p: AR order (autoregressive).
            d: Differencing order.
            q: MA order (moving average).
        """
        self.order = (p, d, q)
        self.model = None
        self._last_date = None
        self._freq = None
    
    def fit(self, data: pd.DataFrame) -> 'ARIMAForecaster':
        # Store date info for predict()
        self._last_date = data.index[-1]
        self._freq = pd.infer_freq(data.index) or 'D'
        
        # Train on values
        self.model = ARIMA(data['value'].values, order=self.order).fit()
        return self
    
    def predict(self, horizon: int) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions
        forecast_values = self.model.forecast(steps=horizon)
        
        # Create future dates
        offset = pd.tseries.frequencies.to_offset(self._freq)

        future_dates = pd.date_range(
            start=self._last_date + offset,
            periods=horizon,
            freq=self._freq
        )
        
        forecast = pd.DataFrame({
            'value': forecast_values
        }, index=future_dates)
    
        forecast.index.name = 'date'
        return forecast 
    
    @staticmethod
    def has_recommendations() -> bool:
        """Override - ARIMA supports parameter recommendations"""
        return True

    @staticmethod
    def get_hyperparameters() -> dict:

        return {
            'p': {
                'label': 'p (AR order)',
                'type': 'int',
                'min': 0,
                'max': 5,
                'default': 1,
                'help': 'How many **past values** to use for prediction'
            },
            'd': {
                'label': 'd (Differencing)',
                'type': 'int',
                'min': 0,
                'max': 2,
                'default': 1,
                'help': 'How many **times the data is differenced** to remove trend'
            },
            'q': {
                'label': 'q (MA order)',
                'type': 'int',
                'min': 0,
                'max': 5,
                'default': 1,
                'help': 'How many **past errors** to use for prediction'
            }
        }
    
    @staticmethod
    def get_recommendations(data: pd.DataFrame) -> dict:
    
        rec_d, d_exp = ARIMAForecaster.recommend_d(data)
        rec_p, p_exp = ARIMAForecaster.recommend_p(data)
        rec_q, q_exp = ARIMAForecaster.recommend_q(data)
        
        return {
            'p': (rec_p, p_exp),
            'd': (rec_d, d_exp),
            'q': (rec_q, q_exp)
        }
    
    @staticmethod
    def recommend_d(data: pd.DataFrame, max_d=2) -> tuple[int, str]:
        """Recommend differencing order using ADF test.
        
        Returns:
            Tuple of (recommended d value, explanation string)."""

        values = data['value']
        
        if len(values.dropna()) < 10:
            return 1, "Not enough data points for analysis. Using d=1 as default."
        
        current_series = values.copy()
        p_value = None
        
        for d in range(max_d + 1):
            try:
                clean_series = current_series.dropna()
                
                # Need enough data points for ADF test
                if len(clean_series) < 10:
                    return 1, "Data too short after differencing. Fallback to d=1."
                
                adf_result = adfuller(clean_series)
                p_value = adf_result[1]
                
                # Check if stationary (p < 0.05 means stationary)
                if p_value < 0.05:
                    if d == 0:
                        explanation = f"Series is already stationary (ADF p-value={p_value:.4f} < 0.05)."
                    elif d == 1:
                        explanation = f"Series became stationary after 1st differencing (ADF p-value={p_value:.4f} < 0.05)."
                    else:
                        explanation = f"Series became stationary after 2nd differencing (ADF p-value={p_value:.4f} < 0.05)."
                    return d, explanation
                
                # Difference for next iteration
                current_series = current_series.diff()
                
            except Exception as e:
                return 1, f"Error in ADF test: {str(e)}. Fallback to d=1."
        
        # Fallback if never became stationary
        return 1, f"Series remains non-stationary (ADF p-value={p_value:.4f} > 0.05). Fallback to d=1."
    
    @staticmethod
    def recommend_p(data: pd.DataFrame, max_lag=20) -> tuple[int, str]:
        """Recommend AR order using PACF analysis.
        
        Returns:
            Tuple of (recommended p value, explanation string)."""
        values = data['value']
        
        clean_data = values.dropna()
        
        if len(clean_data) < 20:
            return 1, "Not enough data points for PACF analysis. Fallback to p=1."
        
        try:
            # Calculate PACF
            pacf_values = pacf(clean_data, nlags=min(max_lag, len(clean_data) // 2))
            confidence = 1.96 / np.sqrt(len(clean_data))
            
            # Find first significant lag
            p = 0
            for lag in range(1, min(len(pacf_values), 6)):
                if abs(pacf_values[lag]) > confidence:
                    p = lag
                else:
                    break  # Stop at first non-significant lag
            
            p = min(p, 5)  # Cap at 5
            
            if p == 0:
                explanation = "No significant PACF values detected. Fallback to p=1."
                p = 1
            else:
                explanation = f"PACF shows {p} significant lag(s) above confidence bound (±{confidence:.3f})."
            
            return p, explanation
            
        except Exception as e:
            return 1, f"Error in PACF analysis: {str(e)}. Fallback to p=1."
    
    @staticmethod
    def recommend_q(data: pd.DataFrame, max_lag=20) -> tuple[int, str]:
        """Recommend MA order using ACF analysis.
        
        Returns:
            Tuple of (recommended q value, explanation string).
        """
        # Handle both DataFrame and Series
        values = data['value']
        
        clean_data = values.dropna()
        
        if len(clean_data) < 20:
            return 1, "Not enough data points for ACF analysis. Fallback to q=1."
        
        try:
            # Calculate ACF
            acf_values = acf(clean_data, nlags=min(max_lag, len(clean_data) // 2))
            confidence = 1.96 / np.sqrt(len(clean_data))
            
            # Find first significant lag
            q = 0
            for lag in range(1, min(len(acf_values), 6)):
                if abs(acf_values[lag]) > confidence:
                    q = lag
                else:
                    break  # Stop at first non-significant lag
            
            q = min(q, 5)  # Cap at 5
            
            if q == 0:
                explanation = "No significant ACF values detected. Fallback to q=1."
                q = 1
            else:
                explanation = f"ACF shows {q} significant lag(s) above confidence bound (±{confidence:.3f})."
            
            return q, explanation
            
        except Exception as e:
            return 1, f"Error in ACF analysis: {str(e)}. Fallback to q=1."