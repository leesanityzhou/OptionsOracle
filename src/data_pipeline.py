"""
Data pipeline module for fetching and processing market data.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List

import pandas as pd

from .data_processors import (
    StockDataFetcher,
    OptionsDataFetcher,
    BaseDataFetcher
)

logger = logging.getLogger(__name__)

class DataPipeline:
    """Pipeline for fetching and processing market data."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the data pipeline.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.stock_fetcher = StockDataFetcher(cache_dir=cache_dir)
        self.options_fetcher = OptionsDataFetcher(cache_dir=cache_dir)
    
    def get_complete_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get complete market data including stock and options data.
        Note: Options data is only available for the current date.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for stock data collection
            end_date: End date for stock data collection
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing:
                - 'stock': Stock price data
                - 'options_calls': Call options data (current date only)
                - 'options_puts': Put options data (current date only)
        """
        # Get stock data
        stock_data = self.stock_fetcher.fetch_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        # Get current options data
        options_calls, options_puts = self.options_fetcher.fetch_data(symbol=symbol)
        
        return {
            'stock': stock_data,
            'options_calls': options_calls,
            'options_puts': options_puts
        }

def get_sp500_symbols() -> List[str]:
    """
    Get list of S&P 500 symbols using pandas_datareader.
    
    Returns:
        List of S&P 500 ticker symbols
    """
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {e}")
        raise
