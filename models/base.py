from abc import ABC, abstractmethod
import pandas as pd

from splitters.base import TimeSeriesSplitter

class ForecastModel(ABC):
    """Base class for all time series forecasting models.
    
    All forecasting models must implement fit() and predict() methods.
    The evaluate() method provides a standard evaluation pipeline using
    train/test splits and metric calculation.
    
    Subclasses should override get_hyperparameters() to define their
    hyperparameters available for UI rendering. 
    
    Optionally subclasses can override has_recommendations() and 
    get_recommendations() to provide intelligent parameter suggestions.
    """
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'ForecastModel':
        """Train the model on historical data.
        
        Args:
            data: DataFrame with DatetimeIndex and 'value' column.
        
        Returns:
            self: For method chaining.
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate forecast for future time steps.
        
        Args:
            horizon: Number of steps to forecast.
        
        Returns:
            DataFrame with DatetimeIndex and 'value' column.
        """
        pass
    
    def evaluate(self, data: pd.DataFrame, splitter: 'TimeSeriesSplitter', horizon: int, metric_functions: dict):
        """Evaluate model using train/test split.
        
        Trains on training portion, evaluates on test portion, then retrains
        on full data to generate final forecast.
        
        Args:
            data: DataFrame with DatetimeIndex and 'value' column.
            splitter: TimeSeriesSplitter instance (e.g., HoldoutPctSplitter).
            horizon: Number of steps to forecast beyond data.
            metric_functions: Dict mapping metric names to functions that
                accept (actual, predicted) and return float.
        
        Returns:
            tuple:Tuple of (metrics, forecast, test_predictions).
        """
        from utils.model_evaluation import calculate_metrics
        
        train_data, test_data = splitter.split(data)
        
        # Train on training data
        self.fit(train_data)
        test_predictions = self.predict(len(test_data))
        
        # Evaluation
        metrics = calculate_metrics(
            actual=test_data['value'],
            predicted=test_predictions['value'],
            metric_functions=metric_functions
        )
        
        # Retrain on full dataset for final forecast
        self.fit(data)
        forecast = self.predict(horizon)
        
        return metrics, forecast, test_predictions

    @staticmethod
    def has_recommendations() -> bool:
        """Check if model supports parameter recommendations.
        
        Returns:            
            bool: True if model provides recommendations, False otherwise.
        """
        return False
    
    @staticmethod
    def get_recommendations(data) -> dict:
        """Generate hyperparameter recommendations based on data analysis.
        
        Args:
            data: DataFrame with DatetimeIndex and 'value' column.  
        Returns:             
            dict: Mapping of hyperparameter names to (value, explanation) tuples.
        """
        return {}
    
    @staticmethod
    def get_hyperparameters() -> dict:
        """Return hyperparameter configuration for UI rendering.
        
        Returns:
            dict: Configuration for each hyperparameter.
        """
        return {}