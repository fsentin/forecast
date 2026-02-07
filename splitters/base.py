from abc import ABC, abstractmethod
import pandas as pd

class TimeSeriesSplitter(ABC):
    """
    Abstract base class for all time series splitters.
    """

    @abstractmethod
    def split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train/test sets.

        Args:
            data: DataFrame with DatetimeIndex and 'value' column.

        Returns:
            List of tuples: [(train_df, test_df), ...] 
            - Can return multiple splits for rolling/expanding splitters.
        """
        pass