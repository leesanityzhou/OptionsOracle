"""
Test module for prepare_training_data script.
"""

import os
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from scripts.prepare_training_data import prepare_training_data

@pytest.fixture
def mock_stock_data():
    """Create mock stock data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    close_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    
    return pd.DataFrame({
        'Close': close_prices,
        'Open': [p - 1 for p in close_prices],
        'High': [p + 1 for p in close_prices],
        'Low': [p - 2 for p in close_prices],
        'Volume': [1000000] * len(close_prices)
    }, index=dates)

@pytest.fixture
def mock_technical_features(mock_stock_data):
    """Create mock technical features."""
    df = pd.DataFrame(index=mock_stock_data.index)
    
    # Generate features with no NaN values
    df['SMA_10'] = mock_stock_data['Close'].rolling(window=3, min_periods=1).mean()
    df['RSI'] = 50 + np.random.randn(len(mock_stock_data)) * 10
    df['MACD'] = 0.1 + np.random.randn(len(mock_stock_data)) * 0.05
    df['MACD_SIGNAL'] = 0.2 + np.random.randn(len(mock_stock_data)) * 0.05
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    df['MOM'] = mock_stock_data['Close'].diff(1).fillna(0)
    df['ROC'] = mock_stock_data['Close'].pct_change().fillna(0) * 100
    df['STOCH_K'] = 50 + np.random.randn(len(mock_stock_data)) * 10
    df['STOCH_D'] = 50 + np.random.randn(len(mock_stock_data)) * 10
    df['WILLR'] = -50 + np.random.randn(len(mock_stock_data)) * 10
    df['SMA_20'] = mock_stock_data['Close'].rolling(window=3, min_periods=1).mean()
    df['SMA_50'] = mock_stock_data['Close'].rolling(window=3, min_periods=1).mean()
    df['EMA_10'] = mock_stock_data['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = mock_stock_data['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = mock_stock_data['Close'].ewm(span=50, adjust=False).mean()
    df['ADX'] = 25 + np.random.randn(len(mock_stock_data)) * 5
    df['CCI'] = np.random.randn(len(mock_stock_data)) * 100
    df['DX'] = 25 + np.random.randn(len(mock_stock_data)) * 5
    df['MINUS_DI'] = 25 + np.random.randn(len(mock_stock_data)) * 5
    df['PLUS_DI'] = 25 + np.random.randn(len(mock_stock_data)) * 5
    df['ATR'] = abs(np.random.randn(len(mock_stock_data))) * 2
    df['NATR'] = abs(np.random.randn(len(mock_stock_data))) * 2
    df['TRANGE'] = abs(np.random.randn(len(mock_stock_data))) * 2
    df['BBANDS_UPPER'] = mock_stock_data['Close'] + np.random.randn(len(mock_stock_data)) * 2
    df['BBANDS_MIDDLE'] = mock_stock_data['Close']
    df['BBANDS_LOWER'] = mock_stock_data['Close'] - np.random.randn(len(mock_stock_data)) * 2
    df['OBV'] = mock_stock_data['Volume'].cumsum()
    df['AD'] = mock_stock_data['Volume'].cumsum()
    df['ADOSC'] = np.random.randn(len(mock_stock_data)) * 100000
    
    # Fill any remaining NaN values
    return df.fillna(0)

def test_prepare_training_data(tmp_path, monkeypatch, mock_stock_data, mock_technical_features):
    """Test prepare_training_data function."""
    # Create temporary output path
    output_path = os.path.join(tmp_path, "test_data.parquet")
    
    # Mock yfinance
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_stock_data
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker
    
    # Mock DataPipeline.get_complete_data
    class MockDataPipeline:
        def get_complete_data(self, symbol, start_date, end_date, use_cache):
            return {'stock': mock_stock_data}
    
    # Mock TechnicalFeatureExtractor.extract_features
    class MockTechnicalExtractor:
        def extract_features(self, data):
            return mock_technical_features
    
    # Mock StockDataFetcher
    class MockStockFetcher:
        def fetch_data(self, symbol, start_date=None, end_date=None, use_cache=True):
            return mock_stock_data
    
    # Apply monkeypatches
    import src.data_pipeline
    import src.feature_extractor.technical_features
    import src.data_processors.stock_data_fetcher
    import yfinance
    monkeypatch.setattr(src.data_pipeline, "DataPipeline", MockDataPipeline)
    monkeypatch.setattr(src.feature_extractor.technical_features, "TechnicalFeatureExtractor", MockTechnicalExtractor)
    monkeypatch.setattr(src.data_processors.stock_data_fetcher, "StockDataFetcher", MockStockFetcher)
    monkeypatch.setattr(yfinance, "Ticker", mock_yf.Ticker)
    
    # Run prepare_training_data
    result_df = prepare_training_data(
        symbol="TEST",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
        output_path=output_path
    )
    
    # Load saved data
    loaded_df = pd.read_parquet(output_path)
    
    # Verify data
    assert len(result_df) == len(mock_stock_data) - 1  # One less due to future returns
    assert result_df.equals(loaded_df)  # Both DataFrames should be identical
    
    # Verify required columns exist with correct data types
    required_columns = ['returns', 'direction', 'volatility', 'RSI', 'MACD', 'SMA_10']
    for col in required_columns:
        assert col in result_df.columns
        assert not result_df[col].isna().any()  # No NaN values
    
    # Verify direction is binary (0 or 1)
    assert set(result_df['direction'].unique()).issubset({0, 1})
    
    # Verify volatility is non-negative
    assert (result_df['volatility'] >= 0).all()
    
    # Verify returns calculation
    close_prices = mock_stock_data['Close']
    expected_returns = ((close_prices - close_prices.shift(1)) / close_prices.shift(1)).shift(-1)[:-1]
    pd.testing.assert_series_equal(
        result_df['returns'],
        expected_returns,
        check_names=False
    ) 