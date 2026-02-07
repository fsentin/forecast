from .holdoutpct import HoldoutPctSplitter
from .base import TimeSeriesSplitter

# Model registry
AVAILABLE_SPLITTERS = {
    'HoldoutPct': HoldoutPctSplitter
}

__all__ = ['TimeSeriesSplitter', 'HoldoutPctSplitter', 'AVAILABLE_SPLITTERS']