"""
Tests for data cache manager.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import time
import os

from src.data_processors.data_cache_manager import DataCacheManager

class TestDataCacheManager(unittest.TestCase):
    """Test cases for data cache manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary cache directory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_manager = DataCacheManager(cache_dir=str(self.temp_dir))
        
        # Create test data
        self.dates = pd.date_range(
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        self.test_data = pd.DataFrame({
            'value': np.random.randn(len(self.dates))
        }, index=self.dates)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_save_and_get_data(self):
        """Test saving and retrieving data from cache."""
        # Save data
        self.cache_manager.save_data('test_key', self.test_data)
        
        # Check cache file exists
        cache_file = self.temp_dir / 'test_key.pkl'
        self.assertTrue(cache_file.exists())
        
        # Get data
        cached_data = self.cache_manager.get_data('test_key')
        pd.testing.assert_frame_equal(self.test_data, cached_data)
    
    def test_cache_staleness(self):
        """Test cache staleness check."""
        # Save data
        self.cache_manager.save_data('test_key', self.test_data)
        cache_file = self.temp_dir / 'test_key.pkl'
        
        # Modify file time to make it stale (2 days old)
        stale_time = time.time() - (2 * 24 * 60 * 60)
        os.utime(cache_file, (stale_time, stale_time))
        
        # Try to get stale data
        cached_data = self.cache_manager.get_data('test_key')
        self.assertIsNone(cached_data)
    
    def test_missing_cache(self):
        """Test handling of missing cache."""
        # Try to get non-existent data
        cached_data = self.cache_manager.get_data('nonexistent_key')
        self.assertIsNone(cached_data)
    
    def test_invalid_cache_data(self):
        """Test handling of invalid cache data."""
        # Create invalid cache file
        cache_file = self.temp_dir / 'invalid_key.pkl'
        cache_file.write_text("invalid data")
        
        # Try to get invalid data
        cached_data = self.cache_manager.get_data('invalid_key')
        self.assertIsNone(cached_data)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Save multiple data files
        keys = ['key1', 'key2', 'key3']
        for key in keys:
            self.cache_manager.save_data(key, self.test_data)
            
        # Clear specific cache
        self.cache_manager.clear_cache('key1')
        self.assertFalse((self.temp_dir / 'key1.pkl').exists())
        self.assertTrue((self.temp_dir / 'key2.pkl').exists())
        self.assertTrue((self.temp_dir / 'key3.pkl').exists())
        
        # Clear all cache
        self.cache_manager.clear_cache()
        self.assertFalse(any(self.temp_dir.glob('*.pkl')))

if __name__ == '__main__':
    unittest.main() 