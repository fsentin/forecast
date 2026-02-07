import pandas as pd
from prophet import Prophet
from .base import ForecastModel


class ProphetForecaster(ForecastModel):
    """Prophet forecasting model using Facebook's Prophet library.
    
    Prophet handles seasonality, holidays, and trend changes automatically.
    Does not provide automatic parameter recommendations.
    """

    def __init__(
        self,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode='additive'
    ) -> None:
        """
        Initialize Prophet model
        
        Args:
            changepoint_prior_scale: Flexibility of trend 
            seasonality_prior_scale: Strength of seasonality 
            seasonality_mode: 'additive' or 'multiplicative'
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        
        self.model = None
        self._last_date = None
        self._freq = None
    
    def fit(self, data: pd.DataFrame) -> 'ProphetForecaster':
        # Store date info for predict()
        self._last_date = data.index[-1]
        self._freq = pd.infer_freq(data.index) or 'D'
        
        # Convert to Prophet format (ds, y)
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data['value'].values
        })
        
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=False,
            weekly_seasonality='auto',
            yearly_seasonality='auto'
        )
        
        # Suppress Prophet's verbose output
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        

        self.model.fit(prophet_df)
        return self
    
    def predict(self, horizon: int) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dates
        future = self.model.make_future_dataframe(periods=horizon, freq=self._freq)
        
        # Get predictions
        forecast = self.model.predict(future)
        
        # Extract only future predictions (last 'horizon' rows)
        forecast_future = forecast.tail(horizon)
        
        # Return as DataFrame with DatetimeIndex and 'value' column
        result_df = pd.DataFrame({
            'value': forecast_future['yhat'].values
        }, index=pd.DatetimeIndex(forecast_future['ds']))
        result_df.index.name = 'date'
    
        return result_df
    
    @staticmethod
    def has_recommendations() -> bool:
        """Override - Prophet does not have statistical recommendations like ARIMA"""
        return False
    
    @staticmethod
    def get_hyperparameters() -> dict:
        return {
            'changepoint_prior_scale': {
                'label': 'Trend Flexibility',
                'type': 'float',
                'min': 0.001,
                'max': 0.5,
                'default': 0.05,
                'step': 0.001,
                'help': 'How flexible the trend is (higher = more flexible, captures more changes)'
            },
            'seasonality_prior_scale': {
                'label': 'Seasonality Strength',
                'type': 'float',
                'min': 0.01,
                'max': 25.0,
                'default': 10.0,
                'step': 0.1,
                'help': 'Strength of seasonal patterns (higher = stronger seasonality)'
            },
            'seasonality_mode': {
                'label': 'Seasonality Mode',
                'type': 'select',
                'options': ['additive', 'multiplicative'],
                'default': 'additive',
                'help': "**Additive**: seasonality constant over time. **Multiplicative**: seasonality scales with trend"
            }
        }