"""
Test utilities for model testing.
"""

import numpy as np
import pandas as pd

def generate_test_features(n_samples: int = 100) -> pd.DataFrame:
    """Generate test feature data."""
    np.random.seed(42)
    
    # Generate technical features
    data = {
        'RSI': np.random.uniform(0, 100, n_samples),
        'MACD': np.random.normal(0, 1, n_samples),
        'MACD_SIGNAL': np.random.normal(0, 1, n_samples),
        'MACD_HIST': np.random.normal(0, 1, n_samples),
        'STOCH_K': np.random.uniform(0, 100, n_samples),
        'STOCH_D': np.random.uniform(0, 100, n_samples),
        'ADX': np.random.uniform(0, 100, n_samples),
        'ATR': np.random.uniform(0, 10, n_samples),
        'BBANDS_UPPER': np.random.uniform(150, 200, n_samples),
        'BBANDS_LOWER': np.random.uniform(100, 150, n_samples)
    }
    
    # Generate options features
    data.update({
        'PUT_CALL_VOLUME_RATIO': np.random.uniform(0.5, 1.5, n_samples),
        'PUT_CALL_OI_RATIO': np.random.uniform(0.5, 1.5, n_samples),
        'IV_SKEW': np.random.normal(0, 0.1, n_samples),
        'DELTA_NEUTRAL_RATIO': np.random.uniform(0.8, 1.2, n_samples),
        'GAMMA_SCALPING_OPPORTUNITY': np.random.uniform(0, 100, n_samples),
        'THETA_VEGA_RATIO': np.random.normal(0, 0.1, n_samples),
        'VEGA_WEIGHTED_CALLS': np.random.uniform(0, 1, n_samples),
        'VEGA_WEIGHTED_PUTS': np.random.uniform(0, 1, n_samples)
    })
    
    return pd.DataFrame(data)

def generate_classification_target(n_samples: int = 100) -> pd.Series:
    """Generate test classification target."""
    np.random.seed(42)
    return pd.Series(
        np.random.choice([-1, 0, 1], size=n_samples),
        name='signal'
    )

def generate_regression_target(n_samples: int = 100) -> pd.Series:
    """Generate test regression target."""
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0, 0.02, n_samples),  # ~2% daily returns
        name='returns'
    )

def generate_sequence_target(n_samples: int = 100) -> pd.DataFrame:
    """Generate test sequence target."""
    np.random.seed(42)
    
    # Generate [returns, volatility, signal]
    data = {
        'returns': np.random.normal(0, 0.02, n_samples),
        'volatility': np.random.uniform(0.1, 0.3, n_samples),
        'signal': np.random.choice([-1, 0, 1], size=n_samples)
    }
    
    return pd.DataFrame(data) 