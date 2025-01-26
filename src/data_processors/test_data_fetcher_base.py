"""
Tests for base data fetcher.
"""

import os
import shutil
import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from .data_fetcher_base import BaseDataFetcher

class MockDataFetcher(BaseDataFetcher):
    """Mock implementation of BaseDataFetcher for testing."""
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Mock implementation of fetch_data."""
        return pd.DataFrame({
            'date': pd.date_range(start=start_date, end=end_date),
            'close': [100.0, 101.0, 102.0]
        })
    
    def _get_cached_data(self, cache_path: str) -> pd.DataFrame:
        """Mock implementation of _get_cached_data."""
        cache_path = str(cache_path)  # Convert PosixPath to string
        if not os.path.exists(cache_path) or not cache_path.endswith('.parquet'):
            return None
        try:
            return pd.read_parquet(cache_path)
        except:
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: str, data_type: str) -> None:
        """Mock implementation of _save_to_cache."""
        cache_path = str(cache_path)  # Convert PosixPath to string
        data.to_parquet(cache_path)
    
    def _is_cache_stale(self, cache_path: str, max_age_days: int = 1) -> bool:
        """Mock implementation of _is_cache_stale."""
        cache_path = str(cache_path)  # Convert PosixPath to string
        if not os.path.exists(cache_path):
            return True
        
        # Get file modification time
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        now = datetime.now()
        
        # Check if file is older than max_age_days
        return (now - mtime).days > max_age_days

class TestDataFetcherBase(unittest.TestCase):
    """Test cases for BaseDataFetcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fetcher = MockDataFetcher(cache_dir=self.temp_dir)
        self.symbol = "AAPL"
        self.test_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', end='2024-01-03'),
            'close': [100.0, 101.0, 102.0]
        })
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 3)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_get_cache_path(self):
        """Test cache path generation."""
        expected_path = os.path.join(self.temp_dir, f"{self.symbol}_data.parquet")
        actual_path = str(self.fetcher._get_cache_path(self.symbol, "data"))
        self.assertEqual(actual_path, expected_path)

    def test_save_and_load_cache(self):
        """Test saving and loading data from cache."""
        cache_path = self.fetcher._get_cache_path(self.symbol, "data")
        self.fetcher._save_to_cache(self.test_data, cache_path, "data")
        loaded_data = self.fetcher._get_cached_data(cache_path)
        pd.testing.assert_frame_equal(loaded_data, self.test_data)

    def test_cache_expiry(self):
        """Test cache expiry check."""
        cache_path = self.fetcher._get_cache_path(self.symbol, "data")
        self.fetcher._save_to_cache(self.test_data, cache_path, "data")
        
        # Cache should be valid initially (less than 1 day old)
        self.assertFalse(self.fetcher._is_cache_stale(cache_path))
        
        # Modify cache file timestamp to make it stale (2 days old)
        old_time = datetime.now() - timedelta(days=2)
        os.utime(str(cache_path), (old_time.timestamp(), old_time.timestamp()))
        self.assertTrue(self.fetcher._is_cache_stale(cache_path))

    def test_invalid_cache_handling(self):
        """Test handling of invalid cache data."""
        cache_path = self.fetcher._get_cache_path(self.symbol, "data")
        
        # Write invalid data to cache
        with open(str(cache_path), 'w') as f:
            f.write('invalid data')
        
        # Should return None for invalid cache
        self.assertIsNone(self.fetcher._get_cached_data(cache_path))

if __name__ == '__main__':
    unittest.main() 