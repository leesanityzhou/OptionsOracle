"""
Data cache manager module.
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

class DataCacheManager:
    """Manages caching of data to disk."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if exists and valid, None otherwise
        """
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            if not cache_file.exists():
                return None
            
            # Check if cache is stale (older than 1 day)
            if self._is_cache_stale(cache_file):
                logger.info(f"Cache for {key} is stale")
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Retrieved {key} from cache")
            return data
            
        except Exception as e:
            logger.warning(f"Error reading cache for {key}: {e}")
            return None
    
    def save_data(self, key: str, data: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {key} to cache")
            
        except Exception as e:
            logger.warning(f"Error saving cache for {key}: {e}")
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Clear cache files.
        
        Args:
            key: Specific cache key to clear, or None to clear all
        """
        try:
            if key:
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    os.remove(cache_file)
                    logger.info(f"Cleared cache for {key}")
            else:
                for file in self.cache_dir.glob("*.pkl"):
                    os.remove(file)
                logger.info("Cleared all cache")
                
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
    
    def _is_cache_stale(self, cache_file: Path) -> bool:
        """Check if cache file is older than 1 day."""
        cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - cache_time
        return age.days >= 1 