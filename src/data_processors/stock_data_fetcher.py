"""
Stock data fetcher module.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .data_fetcher_base import BaseDataFetcher

# Configure logging
logger = logging.getLogger(__name__)

class StockDataFetcher(BaseDataFetcher):
    """Fetches stock data for a given symbol."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the stock data fetcher."""
        super().__init__(cache_dir)
    
    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a given symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame containing historical stock data
        """
        if use_cache:
            cached_data = self._load_from_cache(symbol, "stock")
            if cached_data is not None:
                return cached_data
        
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365))
        
        try:
            # Get stock ticker
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d"
            )
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Basic data cleaning
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            if use_cache:
                self._save_to_cache(data, symbol, "stock")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            raise
