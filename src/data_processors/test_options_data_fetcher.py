"""
Tests for options data fetcher.
"""

import unittest
import os
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch
import pandas as pd

from src.data_processors.options_data_fetcher import OptionsDataFetcher

class TestOptionsDataFetcher(unittest.TestCase):
    """Test cases for options data fetcher."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_cache_dir = "test_cache"
        os.makedirs(self.test_cache_dir, exist_ok=True)
        self.fetcher = OptionsDataFetcher(cache_dir=self.test_cache_dir)
        self.symbol = "AAPL"
        self.current_date = datetime.now()
        
        # Mock stock data
        self.stock_data = pd.DataFrame({
            'Open': [150],
            'High': [155],
            'Low': [145],
            'Close': [152],
            'Volume': [1000000]
        }, index=[self.current_date])
        
        # Mock options data
        self.option_chain = MagicMock()
        self.option_chain.calls = pd.DataFrame({
            'contractSymbol': ['AAPL220101C00150000'],
            'strike': [150],
            'lastPrice': [5],
            'bid': [4.8],
            'ask': [5.2],
            'volume': [100],
            'openInterest': [1000],
            'impliedVolatility': [0.3]
        })
        self.option_chain.puts = pd.DataFrame({
            'contractSymbol': ['AAPL220101P00150000'],
            'strike': [150],
            'lastPrice': [3],
            'bid': [2.8],
            'ask': [3.2],
            'volume': [80],
            'openInterest': [800],
            'impliedVolatility': [0.25]
        })
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    @patch('yfinance.Ticker')
    def test_fetch_data(self, mock_ticker):
        """Test fetching options data."""
        # Mock the ticker
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = self.stock_data
        mock_ticker_instance.options = [self.current_date.strftime('%Y-%m-%d')]
        mock_ticker_instance.option_chain.return_value = self.option_chain
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        calls, puts = self.fetcher.fetch_data(self.symbol)
        
        # Verify data
        self.assertIsInstance(calls, pd.DataFrame)
        self.assertIsInstance(puts, pd.DataFrame)
        self.assertGreater(len(calls), 0)
        self.assertGreater(len(puts), 0)
        
        # Verify columns
        expected_columns = ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'expirationDate', 'daysToExpiry', 'lastTradeDate', 'type']
        self.assertTrue(all(col in calls.columns for col in expected_columns))
        self.assertTrue(all(col in puts.columns for col in expected_columns))
        
        # Verify cache was created
        self.assertTrue(os.path.exists(os.path.join(self.test_cache_dir, f"{self.symbol}_options_calls.parquet")))
        self.assertTrue(os.path.exists(os.path.join(self.test_cache_dir, f"{self.symbol}_options_puts.parquet")))
        
        # Fetch again to test cache
        calls2, puts2 = self.fetcher.fetch_data(self.symbol)
        pd.testing.assert_frame_equal(calls, calls2)
        pd.testing.assert_frame_equal(puts, puts2)
    
    @patch('yfinance.Ticker')
    def test_fetch_data_api_error(self, mock_ticker):
        """Test handling of API errors."""
        # Mock the ticker to raise an exception
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance
        
        # Verify exception is raised
        with self.assertRaises(Exception):
            self.fetcher.fetch_data(self.symbol)
    
    @patch('yfinance.Ticker')
    def test_fetch_data_no_options(self, mock_ticker):
        """Test handling of no options data."""
        # Mock the ticker with stock data but no options
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = self.stock_data
        mock_ticker_instance.options = []  # No options available
        mock_ticker.return_value = mock_ticker_instance
        
        # Verify ValueError is raised
        with self.assertRaises(ValueError):
            self.fetcher.fetch_data(self.symbol)
    
    @patch('yfinance.Ticker')
    def test_fetch_data_no_stock_data(self, mock_ticker):
        """Test handling of missing stock data."""
        # Mock the ticker to return empty stock data
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_ticker_instance
        
        # Verify ValueError is raised
        with self.assertRaises(ValueError):
            self.fetcher.fetch_data(self.symbol)

if __name__ == '__main__':
    unittest.main() 