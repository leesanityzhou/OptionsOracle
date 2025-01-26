"""
Script to prepare training data using technical features.
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.data_pipeline import DataPipeline
from src.feature_extractor.technical_features import TechnicalFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_training_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_path: str,
    force_fetch: bool = False
):
    """
    Prepare training data by fetching market data and extracting technical features.
    
    Args:
        symbol: Stock symbol
        start_date: Start date for data collection
        end_date: End date for data collection
        output_path: Path to save the prepared data
        force_fetch: Whether to force fetch new data instead of using cache
    """
    logger.info(f"Preparing training data for {symbol}")
    
    # Initialize pipeline and feature extractors
    pipeline = DataPipeline()
    technical_extractor = TechnicalFeatureExtractor()
    
    # Fetch data
    logger.info("Fetching market data...")
    data = pipeline.get_complete_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        use_cache=not force_fetch
    )
    
    # Extract technical features
    logger.info("Extracting technical features...")
    features_df = technical_extractor.extract_features(data['stock'])
    
    # Fill NaN values in technical features
    features_df = features_df.fillna(0)
    
    # Calculate labels
    logger.info("Calculating labels...")
    stock_data = data['stock']
    
    # Calculate returns and direction
    close_prices = stock_data['Close']
    features_df['returns'] = (close_prices - close_prices.shift(1)) / close_prices.shift(1)
    features_df['returns'] = features_df['returns'].shift(-1)  # Next day's returns
    features_df['direction'] = (features_df['returns'] > 0).astype(int)
    
    # Calculate volatility using rolling window
    features_df['volatility'] = close_prices.pct_change().rolling(window=20, min_periods=1).std().fillna(0)
    
    # Drop the last row since we can't calculate next day's returns
    features_df = features_df.iloc[:-1]
    
    # Drop rows with NaN values in required columns
    required_columns = ['returns', 'direction', 'volatility']
    features_df = features_df.dropna(subset=required_columns)
    
    # Save to parquet
    logger.info(f"Saving prepared data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.index = pd.to_datetime(features_df.index)
    features_df.index.name = 'date'
    features_df.to_parquet(output_path, index=True)
    
    logger.info(f"Successfully prepared {len(features_df)} samples")
    return features_df

def main():
    """Main function to run data preparation."""
    # Example usage
    symbol = "AAPL"  # Can be changed to any stock symbol
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    output_path = "data/training_data.parquet"
    
    prepare_training_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        force_fetch=True  # Force fetch new data
    )

if __name__ == "__main__":
    main() 