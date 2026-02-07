import streamlit as st
import pandas as pd
from models import AVAILABLE_MODELS
from config.settings import TRAIN_PCT_DEFAULT, HORIZON_DEFAULT


class AppState:
    """Centralized manager for Streamlit session state.
    
    All state access and modifications should go through this 
    interface to maintain clear ownership.
    
    The state includes:
        - Uploaded time series data (historical_data)
        - Model training status and results for each forecasting model
        - Model hyperparameters (train_pct, horizon)
        - Cache invalidation key for uploaded files
        - Preprocessing workflow state
    """
    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables with default values.
        
        This should be called once at app startup. 
        Initializes data storage, model tracking, 
        preprocessing state, and hyperparameters for all available models.
        
        State initialized:
            - historical_data: None (no data loaded yet)
            - model_trained: {} (no models trained yet)
            - upload_key: 0 (cache invalidation counter)
            - preprocessing_pending: False (no preprocessing workflow active)
            - pending_df: None (no data waiting for preprocessing)
            - pending_gap_count: 0 (no gaps detected)
            - Per-model state: train_pct, horizon, results for each model
        """
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = None
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = {}
        if 'upload_key' not in st.session_state:
            st.session_state.upload_key = 0

        if 'preprocessing_pending' not in st.session_state:
            st.session_state.preprocessing_pending = False
        if 'pending_df' not in st.session_state:
            st.session_state.pending_df = None
        if 'pending_gap_count' not in st.session_state:
            st.session_state.pending_gap_count = 0
        
        for model_name in AVAILABLE_MODELS.keys():
            model_key = model_name.lower()
            
            if f'{model_key}_train_pct' not in st.session_state:
                st.session_state[f'{model_key}_train_pct'] = TRAIN_PCT_DEFAULT
            if f'{model_key}_horizon' not in st.session_state:
                st.session_state[f'{model_key}_horizon'] = HORIZON_DEFAULT
            if f'{model_key}_results' not in st.session_state:
                st.session_state[f'{model_key}_results'] = None
            if model_key not in st.session_state.model_trained:
                st.session_state.model_trained[model_key] = False
    
    @staticmethod
    def get_data() -> pd.DataFrame:
        """Get the currently loaded time series dataframe.
        
        Returns:
            DataFrame with DatetimeIndex and 'value' column if data is loaded,
            None if no data has been uploaded yet.
        """
        return st.session_state.get('historical_data')
    
    @staticmethod
    def set_data(historical_data: pd.DataFrame) -> None:
        """Set the time series dataframe and clear all model results.
        
        When new data is loaded, all previously trained models become invalid
        since they were trained on different data. This method clears all model
        results and increments the cache invalidation key.
        
        Args:
            historical_data: DataFrame with DatetimeIndex and 'value' column,
                or None to clear the data.
        """
        st.session_state.historical_data = historical_data
        AppState.clear_all_models()
    
    @staticmethod
    def has_data() -> bool:
        """Check if time series data is currently loaded.
        
        Returns:
            bool: True if data has been loaded and is not None, False otherwise.
        """
        return st.session_state.get('historical_data') is not None
    
    @staticmethod
    def get_model_config(model_name: str):
        """Get the current train/test split and forecast horizon for a model.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Dictionary with keys:
                - 'train_pct': Training percentage (70-90)
                - 'horizon': Forecast horizon in timesteps (1-365)
        """
        key = model_name.lower()
        return {
            'train_pct': st.session_state.get(f'{key}_train_pct'),
            'horizon': st.session_state.get(f'{key}_horizon')
        }
    
    @staticmethod
    def get_model_results(model_name: str) -> dict:
        """Get all training results and metadata for a model.
        
        Args:
            model_name: Name of the model (e.g., 'ARIMA', 'Prophet', 'N-BEATS').
        
        Returns:
            dict: Dictionary with keys:
                - 'trained' (bool): Whether model has been trained
                - 'metrics' (dict): Evaluation metrics (MAE, RMSE, etc.)
                - 'predictions' (DataFrame): Future forecast values
                - 'test_predictions' (DataFrame): Test set predictions for evaluation
                - 'train_pct' (int): Training percentage used during training
                - 'horizon' (int): Forecast horizon used during training
        
        Note:
            All values may be None if model hasn't been trained yet.
        """
        key = model_name.lower()
        return {
            'trained': st.session_state.model_trained.get(key, False),
            'metrics': st.session_state.get(f'{key}_results'),
            'predictions': st.session_state.get(f'{key}_predictions'),
            'test_predictions': st.session_state.get(f'{key}_test_predictions'),
            'train_pct': st.session_state.get(f'{key}_trained_train_pct'),
            'horizon': st.session_state.get(f'{key}_trained_horizon') 
        }
    
    @staticmethod
    def set_model_results(model_name: str, 
                          metrics: dict, 
                          predictions: pd.DataFrame, 
                          test_predictions: pd.DataFrame, 
                          train_pct: int, 
                          horizon: int) -> None:
        """Save model training results and metadata to session state.
        
        This method is called after successful model training to persist all
        results including evaluation metrics, forecasts, and the configuration
        used during training. 
        
        Sets model_trained[model_name] to True, enabling the results view.
        
        Args:
            model_name: Name of the model (e.g., 'ARIMA', 'Prophet', 'N-BEATS').
            metrics: Dictionary of evaluation metrics (e.g., {'mae': 5.2, 'rmse': 7.8}).
            predictions: DataFrame containing future forecast with DatetimeIndex.
            test_predictions: DataFrame containing test set predictions for evaluation.
            train_pct: Training percentage used (stored for comparison validation).
            horizon: Forecast horizon used (stored for reference).        
        """
        key = model_name.lower()
        st.session_state[f'{key}_results'] = metrics
        st.session_state[f'{key}_predictions'] = predictions
        st.session_state[f'{key}_test_predictions'] = test_predictions
        st.session_state[f'{key}_trained_train_pct'] = train_pct
        st.session_state[f'{key}_trained_horizon'] = horizon
        st.session_state.model_trained[key] = True
    
    @staticmethod
    def get_all_trained_models():
        """Get training status for all available models.
        
        Returns:
            Dictionary mapping model names to their training status.
            Example: {'ARIMA': True, 'Prophet': False, 'N-BEATS': True}
        """
        return {
            name: st.session_state.model_trained.get(name.lower(), False)
            for name in AVAILABLE_MODELS.keys()
        }
    
    @staticmethod
    def get_upload_key():
        """Get the current cache invalidation key for file uploads.
        
        Returns:
            Integer that increments each time new data is loaded. Used as
            a parameter to @st.cache_resource to invalidate cached model
            training when data changes.
        """
        return st.session_state.upload_key
    
    @staticmethod
    def clear_all_models():
        """Clear all model training results while preserving user settings.
        
        This is called when new data is loaded to invalidate all cached model
        results, since they were trained on different data. User's hyperparameter
        choices and train/test split settings are preserved.
        
        Side Effects:
            - Resets all model_trained flags to False
            - Clears all metrics, predictions, and test predictions
            - Increments upload_key to invalidate @st.cache_resource
            - Preserves: train_pct, horizon, and hyperparameter settings
        """
        st.session_state.model_trained = {}
        st.session_state.upload_key += 1
        
        for model_name in AVAILABLE_MODELS.keys():
            key = model_name.lower()
            st.session_state.model_trained[key] = False
            st.session_state[f'{key}_results'] = None
            st.session_state[f'{key}_predictions'] = None
            st.session_state[f'{key}_test_predictions'] = None
            st.session_state[f'{key}_trained_train_pct'] = None
            st.session_state[f'{key}_trained_horizon'] = None

    @staticmethod
    def reset_all() -> None:
        """Perform a complete reset to initial application state.
        """
        st.session_state.historical_data = None
        AppState.clear_all_models()
        
        for model_name in AVAILABLE_MODELS.keys():
            key = model_name.lower()
            st.session_state[f'{key}_train_pct'] = TRAIN_PCT_DEFAULT
            st.session_state[f'{key}_horizon'] = HORIZON_DEFAULT

    @staticmethod
    def set_hyperparameter(model_name: str, 
                           param_name: str, 
                           value):
        """Set a model hyperparameter value in session state.
        
        This is used by the "Get Recommendations" feature to programmatically
        apply suggested hyperparameter values.
        
        Args:
            model_name: Name of the model (e.g., 'ARIMA', 'Prophet').
            param_name: Name of the hyperparameter (e.g., 'p', 'd', 'q').
            value: New value for the hyperparameter.

        Note:
            This updates the value that will be used on next model training.
            Does not retrain the model automatically.
        """
        key = f"{model_name.lower()}_{param_name}"
        st.session_state[key] = value

    @staticmethod
    def get_hyperparameter(model_name: str, 
                           param_name: str, 
                           default=None):
        """Get a model hyperparameter value from session state.
        
        Args:
            model_name: Name of the model (e.g., 'ARIMA', 'Prophet').
            param_name: Name of the hyperparameter (e.g., 'p', 'd', 'q').
            default: Value to return if hyperparameter hasn't been set.
        
        Returns:
            Current hyperparameter value, or default if not found.
        """
        key = f"{model_name.lower()}_{param_name}"
        return st.session_state.get(key, default)
    

    @staticmethod
    def set_preprocessing_state(pending: bool, 
                                df: pd.DataFrame = None, 
                                count: int = 0) -> None:
        """Set preprocessing state for gap handling.

        When gaps are detected during data loading, this method stores
        the state needed for the preprocessing workflow: the dataframe with gaps
        and the count of gaps.
        
        Args:
            pending: Whether preprocessing is waiting for user input.
            df: DataFrame with gaps (optional).
            count: Number of gaps found.
        """
        st.session_state.preprocessing_pending = pending
        st.session_state.pending_df = df
        st.session_state.pending_gap_count = count

    @staticmethod
    def get_preprocessing_state() -> tuple[bool, pd.DataFrame, int]:
        """Get current preprocessing state.
        
        Returns:
            tuple: Tuple of (pending, dataframe, gap_count) where:
                - pending: True if preprocessing UI should be shown
                - dataframe: DataFrame with gaps, or None
                - gap_count: Number of gaps detected
        """
        return (
            st.session_state.get('preprocessing_pending', False),
            st.session_state.get('pending_df'),
            st.session_state.get('pending_gap_count', 0)
        )

    @staticmethod
    def clear_preprocessing_state() -> None:
        """Clear preprocessing state after data is loaded.
        
        Called after preprocessing is successfully applied.
        """
        st.session_state.preprocessing_pending = False
        st.session_state.pending_df = None
        st.session_state.pending_gap_count = 0