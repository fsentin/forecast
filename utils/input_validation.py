import pandas as pd
from config.settings import (
    MIN_DATA_POINTS, MAX_DATA_POINTS
)

def validate_date_column(data, column_name) -> tuple[bool, str]:
    """
    Validate that a column contains parseable dates
    
    Returns:
        (is_valid, error_message)
    """
    try:
        col = data[column_name]
        
        # If it's already a datetime dtype, it's valid
        if pd.api.types.is_datetime64_any_dtype(col):
            return True, None
        
        # Reject if column is numeric dtype (int or float)
        if pd.api.types.is_numeric_dtype(col):
            return False, f"'{column_name}' is numeric. Please select a column with date strings such as '2026-01-01'."
        
        # Try parsing all values as dates - must ALL succeed
        pd.to_datetime(col, errors='raise')
        
        return True, None
        
    except Exception as e:
        return False, f"'{column_name}' does not contain valid dates."

def validate_numeric_column(data, column_name) -> tuple[bool, str]:
    """
    Validate that a column contains numeric values.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        pd.to_numeric(data[column_name], errors='raise')
        return True, None
    except Exception as e:
        return False, f"'{column_name}' does not contain valid numbers."

def validate_columns(data, date_column, value_column) -> tuple[bool, str]:
    """
    Validate both date and value columns as text and date, accordingly.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if date_column == value_column:
        errors.append("Date and value columns must be different.")
        return False, errors
    
    date_valid, date_error = validate_date_column(data, date_column)
    if not date_valid:
        errors.append(date_error)
    
    value_valid, value_error = validate_numeric_column(data, value_column)
    if not value_valid:
        errors.append(value_error)
    
    is_valid = len(errors) == 0
    return is_valid, errors

def validate_timeseries_size(
    df: pd.DataFrame,
    min_points: int = MIN_DATA_POINTS,
    max_points: int = MAX_DATA_POINTS
)-> tuple[bool, str]:
    """
    Validate length of times
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []

    n = len(df)

    if n < min_points:
        errors.append(
            f"Time series is too short ({n} points). "
            f"Minimum required is {min_points}."
        )

    if n > max_points:
        errors.append(
            f"Time series is too large ({n} points). "
            f"Maximum allowed is {max_points}."
        )

    return len(errors) == 0, errors
