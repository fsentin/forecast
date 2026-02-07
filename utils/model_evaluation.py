import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

AVAILABLE_METRICS = {
    'mae': {
        'name': 'MAE',
        'function': mean_absolute_error,
        'lower_is_better': True,
        'description': '**Mean Absolute Error**: Average size of the errors in the same units as the data.'
    },
    'rmse': {
        'name': 'RMSE',
        'function': lambda a, p: np.sqrt(mean_squared_error(a, p)),
        'lower_is_better': True,
        'description': '**Root Mean Squared Error**: Square root of the average squared errors. Penalizes larger errors more.'
    }
}

def calculate_metrics(actual, predicted, metric_functions) -> dict:
    """
    Calculate metrics using provided functions.
    
    Args:
        actual: Ground truth values
        predicted: Forecasted values
        metric_functions: Dict of {metric_name: callable_function}
    
    Returns:
        dict: {metric_name: calculated_value}
    """
    results = {}
    for name, func in metric_functions.items():
        try:
            value = func(actual, predicted)
            # Handle inf/nan
            if np.isnan(value) or np.isinf(value):
                results[name] = None
            else:
                results[name] = float(value)
        except Exception as e:
            # Metric calculation failed - store None
            results[name] = None
    
    return results

def set_random_seeds(seed: int) -> None:
    """
    Sets random seed in all relevant libaries.
    
    Args:
        seed: integer seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False