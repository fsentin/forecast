"""Service layer for model training and evaluation workflows."""

import streamlit as st
import pandas as pd
from splitters.holdoutpct import HoldoutPctSplitter
from utils.model_evaluation import AVAILABLE_METRICS


class ModelService:
    """Orchestrates model training, evaluation, and caching."""
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def train_and_evaluate(
        model_class,
        data: pd.DataFrame,
        train_pct: int,
        horizon: int,
        upload_key: str,  # for cache invalidation
        **hyperparams
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        """Train model, evaluate on test set, and generate forecast.
        
        This method is cached based on model class, data, hyperparameters,
        and upload key. Cache invalidates when any of these change.
        
        Args:
            model_class: Model class to instantiate (e.g., ARIMAForecaster).
            data: Historical time series data with DatetimeIndex and 'value' column.
            train_pct: Percentage of data to use for training during evaluation.
            horizon: Number of steps to forecast into the future.
            upload_key: Cache invalidation key that increments on new uploads.
            **hyperparams: Model-specific hyperparameters (e.g., p, d, q for ARIMA).
        
        Returns:
            tuple: (metrics dict, forecast DataFrame, test_predictions DataFrame)
                - metrics: Dictionary of evaluation metrics (MAE, RMSE, etc.)
                - forecast: Future predictions beyond the historical data
                - test_predictions: Predictions on test set for evaluation
        """
        # Instantiate model with hyperparameters
        model = model_class(**hyperparams)
        
        # Prepare evaluation components
        splitter = HoldoutPctSplitter(train_pct=train_pct)
        metric_functions = {
            name: config['function']
            for name, config in AVAILABLE_METRICS.items()
        }
        
        # Execute evaluation pipeline
        metrics, forecast, test_predictions = model.evaluate(
            data=data,
            splitter=splitter,
            horizon=horizon,
            metric_functions=metric_functions
        )
        
        return metrics, forecast, test_predictions
