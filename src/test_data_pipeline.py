"""
Tests for the OptionsOracle data pipeline.
Tests the complete data fetching workflow including stock and options data.
"""

import unittest
from datetime import datetime, timedelta
import sys
from pathlib import Path
import pandas as pd

# Add src to Python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    """Test cases for the complete data pipeline."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.pipeline = DataPipeline()
        self.symbol = "AAPL"
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
    
    def test_fetch_complete_data(self):
        """Test fetching both stock and options data."""
        print(f"\nFetching data for {self.symbol}...")
        
        # Fetch data
        data = self.pipeline.get_complete_data(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verify stock data
        self.assertIsInstance(data['stock'], pd.DataFrame)
        self.assertGreater(len(data['stock']), 0)
        self.assertTrue(all(col in data['stock'].columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
        
        # Print stock data summary
        print("\nStock Data Summary:")
        print(f"Shape: {data['stock'].shape}")
        print(f"Columns: {data['stock'].columns.tolist()}")
        print(f"Date range: {data['stock'].index.min()} to {data['stock'].index.max()}")
        
        # Verify options data (current date only)
        self.assertIsInstance(data['options_calls'], pd.DataFrame)
        self.assertIsInstance(data['options_puts'], pd.DataFrame)
        # Note: We don't check length since options data might be empty for some dates
        if not data['options_calls'].empty:
            self.assertTrue(all(col in data['options_calls'].columns for col in ['strike', 'lastPrice', 'volume', 'openInterest']))
        if not data['options_puts'].empty:
            self.assertTrue(all(col in data['options_puts'].columns for col in ['strike', 'lastPrice', 'volume', 'openInterest']))
        
        # Print options data summary
        print(f"\nOptions Data Summary:")
        print(f"Number of call options: {len(data['options_calls'])}")
        print(f"Call options columns: {data['options_calls'].columns.tolist()}")
        print(f"\nNumber of put options: {len(data['options_puts'])}")
        print(f"Put options columns: {data['options_puts'].columns.tolist()}")
        
        # Verify options data structure
        expected_options_columns = {'strike', 'lastTradeDate', 'expirationDate', 'daysToExpiry'}
        self.assertTrue(expected_options_columns.issubset(data['options_calls'].columns))
        self.assertTrue(expected_options_columns.issubset(data['options_puts'].columns))

if __name__ == '__main__':
    unittest.main(verbosity=2) 