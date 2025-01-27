"""
Prepare training data for stock price prediction.
"""

import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
import ta
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    """
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
        
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.
    """
    logger.info("Adding technical indicators")
    
    # Price indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # Momentum indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    
    # Volatility indicators
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Volume indicators
    df['Volume_SMA'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    
    return df

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and labels for training.
    
    Returns:
        Tuple of (features, labels)
    """
    logger.info("Preparing features and labels")
    
    # Normalize price and volume data
    price_scaler = df['Close'].mean()
    volume_scaler = df['Volume'].mean()
    
    # Create features
    features = pd.DataFrame()
    
    # Price features (normalized)
    features['price'] = df['Close'] / price_scaler
    features['high'] = df['High'] / price_scaler
    features['low'] = df['Low'] / price_scaler
    features['open'] = df['Open'] / price_scaler
    
    # Volume features (normalized)
    features['volume'] = df['Volume'] / volume_scaler
    features['volume_sma'] = df['Volume_SMA'] / volume_scaler
    
    # Technical indicators
    features['sma_20'] = df['SMA_20'] / price_scaler
    features['sma_50'] = df['SMA_50'] / price_scaler
    features['ema_20'] = df['EMA_20'] / price_scaler
    features['rsi'] = df['RSI'] / 100  # RSI is already 0-100
    features['macd'] = df['MACD']
    features['macd_signal'] = df['MACD_Signal']
    features['bb_high'] = df['BB_High'] / price_scaler
    features['bb_low'] = df['BB_Low'] / price_scaler
    features['atr'] = df['ATR'] / price_scaler
    
    # Create labels (next day's closing price)
    labels = df['Close'].shift(-1) / price_scaler
    
    # Drop rows with NaN values
    features = features.dropna()
    labels = labels[features.index]
    
    # Remove the last row since we don't have next day's price
    features = features[:-1]
    labels = labels[:-1]
    
    # Convert to tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.float32)
    
    return features_tensor, labels_tensor, price_scaler

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for stock price prediction")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument("--days", type=int, default=365, help="Number of days of historical data")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for training data")
    parser.add_argument("--force_fetch", action="store_true", help="Force fetch new data")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Fetch and process data
    df = fetch_stock_data(args.symbol, start_date, end_date)
    df = add_technical_indicators(df)
    features, labels, price_scaler = prepare_features(df)
    
    logger.info(f"Prepared {len(features)} samples")
    
    # Save data
    torch.save({
        'features': features,
        'labels': labels,
        'price_scaler': price_scaler,
        'symbol': args.symbol,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    }, args.output_path)
    
    logger.info(f"Saved prepared data to {args.output_path}")

if __name__ == "__main__":
    main() 