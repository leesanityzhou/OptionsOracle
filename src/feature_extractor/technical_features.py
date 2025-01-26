"""
Technical feature extractor for stock price data.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import talib

from .base_extractor import BaseFeatureExtractor
from ..utils.config import get_feature_groups

# Configure logging
logger = logging.getLogger(__name__)

class TechnicalFeatureExtractor(BaseFeatureExtractor):
    """Extract technical analysis features from stock price data."""
    
    def __init__(
        self,
        feature_groups: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the technical feature extractor.
        
        Args:
            feature_groups: List of feature groups to extract (from FEATURE_GROUPS)
            feature_names: List of specific features to extract
        """
        super().__init__(feature_names)
        
        # Load feature groups from config
        self.FEATURE_GROUPS = get_feature_groups('technical')
        
        # Set feature groups to extract
        self.feature_groups = feature_groups or list(self.FEATURE_GROUPS.keys())
        
        # Validate feature groups
        invalid_groups = [g for g in self.feature_groups if g not in self.FEATURE_GROUPS]
        if invalid_groups:
            raise ValueError(f"Invalid feature groups: {invalid_groups}")
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical features from stock price data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        try:
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            self._validate_data(data, required_cols)
            
            # Initialize features DataFrame
            features = pd.DataFrame(index=data.index)
            
            # Extract features by group
            if 'momentum' in self.feature_groups:
                momentum_features = self._extract_momentum_features(data)
                features = pd.concat([features, momentum_features], axis=1)
            
            if 'trend' in self.feature_groups:
                trend_features = self._extract_trend_features(data)
                features = pd.concat([features, trend_features], axis=1)
            
            if 'volatility' in self.feature_groups:
                volatility_features = self._extract_volatility_features(data)
                features = pd.concat([features, volatility_features], axis=1)
            
            if 'volume' in self.feature_groups:
                volume_features = self._extract_volume_features(data)
                features = pd.concat([features, volume_features], axis=1)
            
            # Filter features if specific ones are requested
            if self.feature_names:
                features = features[self.feature_names]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            raise
    
    def _extract_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum indicators."""
        df = pd.DataFrame(index=data.index)
        
        # RSI
        df['RSI'] = talib.RSI(data['Close'])
        
        # MACD
        macd, signal, hist = talib.MACD(data['Close'])
        df['MACD'] = macd
        df['MACD_SIGNAL'] = signal
        df['MACD_HIST'] = hist
        
        # Momentum
        df['MOM'] = talib.MOM(data['Close'])
        
        # Rate of Change
        df['ROC'] = talib.ROC(data['Close'])
        
        # Stochastic
        slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'])
        df['STOCH_K'] = slowk
        df['STOCH_D'] = slowd
        
        # Williams %R
        df['WILLR'] = talib.WILLR(data['High'], data['Low'], data['Close'])
        
        return df
    
    def _extract_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract trend indicators."""
        df = pd.DataFrame(index=data.index)
        
        # Moving Averages
        df['SMA_10'] = talib.SMA(data['Close'], timeperiod=10)
        df['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        
        df['EMA_10'] = talib.EMA(data['Close'], timeperiod=10)
        df['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(data['Close'], timeperiod=50)
        
        # ADX
        df['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
        
        # CCI
        df['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'])
        
        # DX
        df['DX'] = talib.DX(data['High'], data['Low'], data['Close'])
        
        # DI
        df['MINUS_DI'] = talib.MINUS_DI(data['High'], data['Low'], data['Close'])
        df['PLUS_DI'] = talib.PLUS_DI(data['High'], data['Low'], data['Close'])
        
        return df
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility indicators."""
        df = pd.DataFrame(index=data.index)
        
        # ATR
        df['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
        df['NATR'] = talib.NATR(data['High'], data['Low'], data['Close'])
        df['TRANGE'] = talib.TRANGE(data['High'], data['Low'], data['Close'])
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'])
        df['BBANDS_UPPER'] = upper
        df['BBANDS_MIDDLE'] = middle
        df['BBANDS_LOWER'] = lower
        
        return df
    
    def _extract_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume indicators."""
        df = pd.DataFrame(index=data.index)
        
        # OBV
        df['OBV'] = talib.OBV(data['Close'], data['Volume'])
        
        # A/D Line
        df['AD'] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
        
        # Chaikin A/D Oscillator
        df['ADOSC'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'])
        
        return df 