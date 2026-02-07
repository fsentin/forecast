import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config.settings import RANDOM_SEED
from .base import ForecastModel


class NBEATSForecaster(ForecastModel):
    """N-BEATS neural forecasting model using Darts library.
    
    N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep learning
    architecture designed specifically for univariate time series forecasting.
    
    This implementation uses a lightweight architecture optimized for speed.
    Supports multiple data scaling strategies:
    - Standard (z-score): Recommended. Less sensitive to outliers.
    - MinMax [0,1]: Bounds to [0,1] but sensitive to outliers.
    - None: No scaling (neural networks typically fail).
    
    When scaling is enabled, predictions are automatically inverse-transformed 
    back to original scale.
    
    Does not provide automatic parameter recommendations.
    """

    def __init__(
        self,
        input_chunk_length=5,
        n_epochs=10,
        scaler_type='standard',
        random_state=RANDOM_SEED
    ) -> None:
        """
        Initialize N-BEATS model (tiny configuration, optimized for speed)
        
        Args:
            input_chunk_length: Number of past time steps to use (lookback window)
            n_epochs: Number of training epochs
            scaler_type: Scaling method - 'Standard (z-score)', 'MinMax (0-1)', 'None',
                        or lowercase variants 'standard', 'minmax', 'none'
            random_state: Random seed for reproducibility
        """
        self.input_chunk_length = input_chunk_length
        self.n_epochs = n_epochs
        
        # Map display names to internal values
        scaler_mapping = {
            'Standard (z-score)': 'standard',
            'MinMax (0-1)': 'minmax',
            'None': 'none',
            'standard': 'standard',
            'minmax': 'minmax',
            'none': 'none'
        }
        self.scaler_type = scaler_mapping.get(scaler_type, scaler_type.lower())
        self.random_state = random_state
        
        # Model components - initialize appropriate scaler
        self.model = None
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:  # 'none'
            self.scaler = None
        
        # Metadata for prediction
        self._last_date = None
        self._freq = None
        self._timeseries = None
        self._output_chunk_length = None
    
    def fit(self, data: pd.DataFrame) -> 'NBEATSForecaster':
        """Train N-BEATS on historical data with optional scaling.
        
        If a scaler is configured, data is scaled before training and the scaler
        is stored for inverse transformation during prediction. If no scaler is
        used (scaler_type='none'), raw data is used directly (not recommended - 
        neural networks are sensitive to scale!).
        
        Args:
            data: DataFrame with DatetimeIndex and 'value' column.
        
        Returns:
            self: For method chaining.
        """
        # Store date info for predict()
        self._last_date = data.index[-1]
        self._freq = pd.infer_freq(data.index) or 'D'
        
        # Optionally scale data
        if self.scaler is not None:
            values = data['value'].values.reshape(-1, 1)
            scaled_values = self.scaler.fit_transform(values)
            
            # Create scaled DataFrame for Darts
            processed_data = pd.DataFrame({
                'value': scaled_values.flatten()
            }, index=data.index)
        else:
            # Use raw data (not recommended - may cause training instability!)
            processed_data = data.copy()
        
        # Convert to Darts TimeSeries
        self._timeseries = TimeSeries.from_dataframe(
            processed_data,
            value_cols='value',
            fill_missing_dates=True,
            freq=self._freq
        )
        
        # Set output_chunk_length to match input (simpler)
        self._output_chunk_length = self.input_chunk_length
        
        # Initialize N-BEATS model with Tiny architecture
        self.model = NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self._output_chunk_length,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            batch_size=32,  # Larger batch = faster training
            model_name="NBEATS_Tiny",
            force_reset=True,
            save_checkpoints=False,
            pl_trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "accelerator": "cpu",  # Force CPU to avoid GPU overhead
                "enable_checkpointing": False
            }
        )
        
        # Train the model
        self.model.fit(self._timeseries, verbose=False)
        
        return self
    
    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate forecast with optional inverse scaling to original scale.
        
        If a scaler was used during training, predictions are inverse-transformed 
        back to original scale. If no scaler was used, raw predictions are returned.
        
        Args:
            horizon: Number of steps to forecast.
        
        Returns:
            DataFrame with DatetimeIndex and 'value' column.
        """
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate predictions
        forecast_ts = self.model.predict(n=horizon, verbose=False)
        
        # Extract values
        predictions = forecast_ts.values().flatten()
        
        # Optionally inverse transform to original scale
        if self.scaler is not None:
            predictions = predictions.reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            predictions = predictions.flatten()
        
        # Convert back to pandas DataFrame
        forecast_df = pd.DataFrame({
            'value': predictions
        }, index=forecast_ts.time_index)
        forecast_df.index.name = 'date'
        return forecast_df
    
    @staticmethod
    def has_recommendations():
        """Override - N-BEATS does not have statistical recommendations like ARIMA"""
        return False
    
    @staticmethod
    def get_hyperparameters():
        return {
            'input_chunk_length': {
                'label': 'Lookback Window',
                'type': 'int',
                'min': 3,
                'max': 30,
                'default': 5,
                'help': 'Number of **past timesteps** to look at (smaller = faster)'
            },
            'n_epochs': {
                'label': 'Training Epochs',
                'type': 'int',
                'min': 10,
                'max': 150,
                'default': 10,
                'help': 'Number of **training iterations** (fewer = faster)'
            },
            'scaler_type': {
                'label': 'Data Scaling',
                'type': 'select',
                'options': ['Standard (z-score)', 'MinMax (0-1)', 'None'],
                'default': 'Standard (z-score)',
                'help': '**Standard (recommended)**: Scales to mean=0, std=1. Less sensitive to outliers. '
                        '**MinMax**: Scales to [0,1] range, but outliers can squish other values near zero. '
                        '**None**: No scaling - neural networks typically fail without it (predictions collapse). '
                        'Try all three to see how preprocessing affects deep learning!'
            }
        }