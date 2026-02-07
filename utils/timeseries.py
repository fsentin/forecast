from datetime import datetime
import pandas as pd

def format_duration(start_date: datetime | pd.Timestamp, 
                    end_date: datetime | pd.Timestamp) -> str:
    """
    Format duration between two dates in human-readable format.
    
    Args:
        start_date: datetime or pd.Timestamp
        end_date: datetime or pd.Timestamp
    
    Returns:
        str: Formatted duration (e.g., "373 days", "13 months", "2 years")
    """
    duration_days = (end_date - start_date).days
    
    if duration_days >= 365:
        years = int(duration_days / 365.25)
        return f"{years} year" if years == 1 else f"{years} years"
    elif duration_days >= 30:
        months = int(duration_days / 30.44)
        return f"{months} month" if months == 1 else f"{months} months"
    else:
        return f"{duration_days} day" if duration_days == 1 else f"{duration_days} days"
    
def is_equally_spaced(df: pd.DataFrame) -> tuple[bool, str, int]:
    """Check if time series has equally spaced data points.
    
    Args:
        df: DataFrame with DatetimeIndex and 'value' column.
    
    Returns:
        Tuple of (is_equal: bool, frequency: str, gaps_count: int).
    """
    if len(df) < 2:
        return True, None, 0
    
    # Infer frequency
    freq = pd.infer_freq(df.index)
    
    # If can't infer (has gaps), guess from most common difference
    if freq is None:
        diffs = df.index.to_series().diff().dropna()
        most_common_diff = diffs.mode()[0]
        
        # Map common differences to frequency strings
        if most_common_diff == pd.Timedelta(days=1):
            freq = 'D'
        elif most_common_diff == pd.Timedelta(days=7):
            freq = 'W'
        elif most_common_diff >= pd.Timedelta(days=28) and most_common_diff <= pd.Timedelta(days=31):
            freq = 'MS'  # Month start
        elif most_common_diff == pd.Timedelta(hours=1):
            freq = 'H'
        else:
            # Can't determine frequency
            return False, None, 0
    
    # Create expected date range
    expected_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )
    
    # Count missing dates
    gaps = expected_range.difference(df.index)
    gaps_count = len(gaps)
    
    is_equal = gaps_count == 0
    
    return is_equal, freq, gaps_count


def fill_gaps_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dates using linear interpolation.
    
    Args:
        df: DataFrame with DatetimeIndex and 'value' column.
    
    Returns:
        DataFrame with filled dates.
    """
    df = df.copy()
    
    # Infer frequency
    freq = pd.infer_freq(df.index) or 'D'
    
    # Create complete date range
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )
    
    # Reindex and interpolate
    df = df.reindex(full_range)
    df['value'] = df['value'].interpolate(method='linear')
    df.index.name = 'date'
    
    return df


def fill_gaps_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dates with zeros.
    
    Args:
        df: DataFrame with DatetimeIndex and 'value' column.
    
    Returns:
        DataFrame with filled dates.
    """
    df = df.copy()
    
    # Infer frequency
    freq = pd.infer_freq(df.index) or 'D'
    
    # Create complete date range
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )
    
    # Reindex and fill with zeros
    df = df.reindex(full_range)
    df['value'] = df['value'].fillna(0)
    df.index.name = 'date'
    
    return df