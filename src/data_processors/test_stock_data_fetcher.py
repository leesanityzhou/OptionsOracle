"""
Tests for stock data fetcher.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile

from src.data_processors.stock_data_fetcher import StockDataFetcher

class TestStockDataFetcher(unittest.TestCase):
    """Test cases for stock data fetcher."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fetcher = StockDataFetcher(cache_dir=str(self.temp_dir))
        self.symbol = "AAPL"
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        
        # Create sample data with today's date
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(150, 200, 1),
            'High': np.random.uniform(160, 210, 1),
            'Low': np.random.uniform(140, 190, 1),
            'Close': np.random.uniform(145, 205, 1),
            'Volume': np.random.randint(1000000, 10000000, 1),
            'Dividends': np.zeros(1),
            'Stock Splits': np.zeros(1)
        }, index=[pd.Timestamp.now().normalize()])
        
        # Ensure High >= Open, Close, Low and Low <= Open, Close
        self.sample_data['High'] = self.sample_data[['Open', 'Close', 'High']].max(axis=1)
        self.sample_data['Low'] = self.sample_data[['Open', 'Close', 'Low']].min(axis=1)
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('yfinance.Ticker')
    def test_fetch_data(self, mock_ticker):
        """Test fetching stock data."""
        # Mock the ticker
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        data = self.fetcher.fetch_data(
            self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verify ticker was called correctly
        mock_ticker.assert_called_once_with(self.symbol)
        mock_ticker_instance.history.assert_called_once_with(
            start=self.start_date,
            end=self.end_date,
            interval="1d"
        )
        
        # Verify data
        pd.testing.assert_frame_equal(data, self.sample_data)
    
    @patch('yfinance.Ticker')
    def test_fetch_data_with_cache(self, mock_ticker):
        """Test fetching data with cache."""
        # Mock the ticker
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        # First fetch should use API
        data1 = self.fetcher.fetch_data(
            self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Second fetch should use cache
        data2 = self.fetcher.fetch_data(
            self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verify ticker was called only once
        mock_ticker.assert_called_once_with(self.symbol)
        
        # Verify both results are the same
        pd.testing.assert_frame_equal(data1, data2)
    
    @patch('yfinance.Ticker')
    def test_fetch_data_no_cache(self, mock_ticker):
        """Test fetching data without cache."""
        # Mock the ticker
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data without cache
        data = self.fetcher.fetch_data(
            self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            use_cache=False
        )
        
        # Verify ticker was called
        mock_ticker.assert_called_once_with(self.symbol)
        
        # Verify data
        pd.testing.assert_frame_equal(data, self.sample_data)
        
        # Verify no cache file was created
        cache_file = self.temp_dir / f"{self.symbol}_stock.parquet"
        self.assertFalse(cache_file.exists())
    
    @patch('yfinance.Ticker')
    def test_fetch_data_empty_response(self, mock_ticker):
        """Test handling of empty response."""
        # Mock the ticker to return empty data
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Verify ValueError is raised
        with self.assertRaises(ValueError):
            self.fetcher.fetch_data(
                self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
    
    @patch('yfinance.Ticker')
    def test_fetch_data_api_error(self, mock_ticker):
        """Test handling of API error."""
        # Mock the ticker to raise an exception
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance
        
        # Verify exception is propagated
        with self.assertRaises(Exception):
            self.fetcher.fetch_data(
                self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )

if __name__ == '__main__':
    unittest.main() 