"""
Base data fetcher module.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseDataFetcher(ABC):
    """Base class for data fetchers."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the base data fetcher.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, symbol: str, data_type: str) -> Path:
        """Get the cache file path for a given symbol and data type."""
        return self.cache_dir / f"{symbol}_{data_type}.parquet"
    
    def _load_from_cache(self, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        cache_path = self._get_cache_path(symbol, data_type)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                # Check if cache is from today
                if datetime.now().date() == pd.Timestamp(df.index[-1]).date():
                    logger.info(f"Loaded {symbol} {data_type} data from cache")
                    return df
            except Exception as e:
                logger.warning(f"Error loading cache for {symbol}: {e}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, data_type: str) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol, data_type)
        df.to_parquet(cache_path)
        logger.info(f"Cached {symbol} {data_type} data")

    @abstractmethod
    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch data for a given symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame containing fetched data
        """
        pass