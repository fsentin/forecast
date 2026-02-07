"""Service layer for data loading and preprocessing workflows."""

import pandas as pd
import numpy as np
from utils.timeseries import (
    is_equally_spaced,
    fill_gaps_interpolate,
    fill_gaps_zero
)

class DataService:
    """Handles data loading, validation, and preprocessing."""
    
    @staticmethod
    def prepare_dataframe(
        data: pd.DataFrame,
        date_column: str,
        value_column: str
    ) -> pd.DataFrame:
        """Convert raw uploaded data to standardized time series format.
        
        Takes a raw CSV dataframe and transforms it into the standard format
        expected by forecasting models: DatetimeIndex with 'value' column.
        
        Args:
            data: Raw uploaded dataframe with arbitrary columns.
            date_column: Name of the column containing date/time values.
            value_column: Name of the column containing numeric values to forecast.
        
        Returns:
            DataFrame with DatetimeIndex and 'value' column, sorted by date.
        
        Raises:
            ValueError: If date column cannot be parsed as datetime.
        """
        # Extract relevant columns
        df = data[[date_column, value_column]].copy()
        df.columns = ['date', 'value']
        
        # Convert to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        return df
    
    @staticmethod
    def check_gaps(df: pd.DataFrame) -> tuple[bool, int]:
        """Check if time series has gaps.
        
        Uses frequency detection to identify gaps in the time series
        that would require preprocessing before modeling.
        
        Args:
            df: DataFrame with DatetimeIndex.
        
        Returns:
            tuple: (has_gaps_dates, missing_count)
                - has_gaps: True if gaps detected
                - gaps_count: Number of gaps 
        """
        is_equal, freq, gaps_count = is_equally_spaced(df)
        has_missing = not is_equal and gaps_count > 0
        return has_missing, gaps_count
    
    @staticmethod
    def apply_preprocessing(
        df: pd.DataFrame,
        method: str
    ) -> pd.DataFrame:
        """Apply preprocessing method to fill gaps.
        
        Args:
            df: DataFrame with gaps.
            method: Preprocessing method - either 'interpolate' or 'zero'.
        
        Returns:
            DataFrame with gaps filled according to method.
        
        Raises:
            ValueError: If method is not recognized.
        """
        if method == "interpolate":
            return fill_gaps_interpolate(df)
        elif method == "zero":
            return fill_gaps_zero(df)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    @staticmethod
    def detect_and_remove_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Detect and remove outliers using IQR method, replacing with interpolation.
        
        Uses the Interquartile Range (IQR) method to identify outliers:
        - Outliers are values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        - Outliers are replaced with interpolated values
        - First/last points are removed if outliers
        
        Args:
            df: DataFrame with DatetimeIndex and 'value' column.
        
        Returns:
            tuple: (cleaned_df, outlier_count)
                - cleaned_df: DataFrame with outliers replaced
                - outlier_count: Number of outliers detected and replaced
        """
        df_clean = df.copy()
        values = df_clean['value']
        
        # Calculate IQR bounds
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            # Remove rows with outliers (their dates are now "missing")
            df_clean = df_clean.loc[~outlier_mask]

            # Reuse your fill_gaps_interpolate function to fill these "missing" dates
            df_clean = fill_gaps_interpolate(df_clean)
        
        return df_clean, outlier_count