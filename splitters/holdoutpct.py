import pandas as pd
from .base import TimeSeriesSplitter

class HoldoutPctSplitter(TimeSeriesSplitter):
    """Split time series data into train/test sets using percentage-based holdout.
    
    Performs a chronological split where the first train_pct% of data is used 
    for training and the remaining data for testing. Ensures at least one data 
    point in both train and test sets to prevent edge cases.
    
    Respects temporal ordering and does not shuffle data.
    
    Attributes:
        train_pct (int): Percentage of data to use for training (1-99).
    """
    
    def __init__(self, train_pct=80) -> None:
        if not 0 < train_pct < 100: 
            raise ValueError()
        self.train_pct = train_pct

    def split(self, data) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into chronological train and test sets.
        
        The split point is calculated as train_pct% of the total data length,
        with safeguards to ensure both sets have at least one data point.
        
        Args:
            data (pd.DataFrame): Time series data with DatetimeIndex and 'value' column.
        
        Returns:
            tuple: A tuple of (train_data, test_data) 
                
        Note:
            For very small datasets (n < 3), the split may not exactly match 
            train_pct to ensure both sets contain at least one point.
        """
        n = len(data)

        # ensure at least 1 point in train and 1 point in test
        split_idx = max(1, min(n - 1, int(n * self.train_pct / 100)))
        
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        return train, test