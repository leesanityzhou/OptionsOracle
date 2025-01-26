"""
Options data fetcher module using yfinance for current options data.
"""

import logging
from datetime import datetime
from typing import Tuple
import os
import pandas as pd
import yfinance as yf

from .data_fetcher_base import BaseDataFetcher

logger = logging.getLogger(__name__)

class OptionsDataFetcher(BaseDataFetcher):
    """Fetches current options data for a given symbol using yfinance."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the options data fetcher.
        
        Args:
            cache_dir: Directory to store cached data
        """
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol: str, data_type: str) -> str:
        """Get the cache file path for the given symbol and data type.
        
        Args:
            symbol: Stock symbol
            data_type: Type of data ('calls' or 'puts')
            
        Returns:
            Path to the cache file
        """
        return os.path.join(self.cache_dir, f"{symbol}_options_{data_type}.parquet")
    
    def fetch_data(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch current options chain data for a given symbol.
        Note: yfinance only provides current options data, not historical.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (calls_df, puts_df) containing current options chain data
            
        Raises:
            ValueError: If no options data is available or if stock data is missing
            Exception: If there is an error fetching data from the API
        """
        try:
            # Check cache first
            calls_cache_path = self._get_cache_path(symbol, 'calls')
            puts_cache_path = self._get_cache_path(symbol, 'puts')
            
            if os.path.exists(calls_cache_path) and os.path.exists(puts_cache_path):
                try:
                    calls_df = pd.read_parquet(calls_cache_path)
                    puts_df = pd.read_parquet(puts_cache_path)
                    
                    # Check if cache is from today
                    if not calls_df.empty and not puts_df.empty:
                        cache_date = pd.to_datetime(calls_df['lastTradeDate'].iloc[0]).date()
                        if cache_date == datetime.now().date():
                            return calls_df, puts_df
                except Exception as e:
                    logger.warning(f"Error reading cache for {symbol}: {str(e)}")
            
            # Get stock ticker
            ticker = yf.Ticker(symbol)
            
            # Get stock data to verify it exists
            stock_data = ticker.history(period='1d')
            if stock_data.empty:
                logger.error(f"No stock data available for {symbol}")
                raise ValueError(f"No stock data available for {symbol}")
            
            # Get all expiry dates
            expiry_dates = ticker.options
            if not expiry_dates:
                logger.warning(f"No options data available for {symbol}")
                raise ValueError(f"No options data available for {symbol}")
            
            # Initialize empty DataFrames
            all_calls = pd.DataFrame()
            all_puts = pd.DataFrame()
            
            # Current date
            current_date = datetime.now()
            
            # Fetch data for each expiry date
            for expiry in expiry_dates:
                try:
                    chain = ticker.option_chain(expiry)
                    expiry_date = pd.to_datetime(expiry)
                    days_to_expiry = (expiry_date - current_date).days
                    
                    # Process calls
                    calls = chain.calls.copy()
                    calls['expirationDate'] = expiry_date
                    calls['daysToExpiry'] = days_to_expiry
                    calls['lastTradeDate'] = current_date
                    calls['type'] = 'call'
                    
                    # Process puts
                    puts = chain.puts.copy()
                    puts['expirationDate'] = expiry_date
                    puts['daysToExpiry'] = days_to_expiry
                    puts['lastTradeDate'] = current_date
                    puts['type'] = 'put'
                    
                    # Append to main DataFrames
                    all_calls = pd.concat([all_calls, calls], ignore_index=True) if not all_calls.empty else calls
                    all_puts = pd.concat([all_puts, puts], ignore_index=True) if not all_puts.empty else puts
                    
                except Exception as e:
                    logger.warning(f"Error fetching options chain for expiry {expiry}: {str(e)}")
                    continue
            
            # Cache the data
            if not all_calls.empty:
                all_calls.to_parquet(calls_cache_path)
            if not all_puts.empty:
                all_puts.to_parquet(puts_cache_path)
            
            return all_calls, all_puts
            
        except ValueError as e:
            logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            raise 