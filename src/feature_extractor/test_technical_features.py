"""
Tests for technical feature extraction methods.
"""

import unittest
import numpy as np
import pandas as pd

from src.feature_extractor import TechnicalFeatureExtractor

class TestTechnicalFeatures(unittest.TestCase):
    """Test cases for technical feature extraction methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = TechnicalFeatureExtractor()
        
        # Create sample OHLCV data with realistic price movements
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        
        # Generate realistic price series
        close = 100 * np.exp(np.random.normal(0.0002, 0.02, 100).cumsum())
        high = close * (1 + np.abs(np.random.normal(0, 0.02, 100)))
        low = close * (1 - np.abs(np.random.normal(0, 0.02, 100)))
        open_price = close * (1 + np.random.normal(0, 0.02, 100))
        volume = np.random.uniform(1000000, 5000000, 100)
        
        self.data = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        # Ensure High is highest and Low is lowest for each row
        self.data['High'] = self.data[['Open', 'High', 'Close']].max(axis=1)
        self.data['Low'] = self.data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Handle any remaining inconsistencies
        self.data = self._validate_ohlc(self.data)
    
    def _validate_ohlc(self, df):
        """Validate and fix OHLC data."""
        # Ensure all values are positive
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = np.abs(df[col])
        
        # Ensure High >= Open, Close
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        
        # Ensure Low <= Open, Close
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        
        # Ensure Volume is positive
        df['Volume'] = np.abs(df['Volume'])
        
        return df
    
    def test_momentum_features(self):
        """Test momentum feature extraction."""
        features = self.extractor._extract_momentum_features(self.data)
        
        # Check all momentum features are present
        expected_features = {
            'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'MOM', 'ROC', 'STOCH_K', 'STOCH_D', 'WILLR'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate RSI bounds (allow for NaN at start)
        rsi = features['RSI'].dropna()
        self.assertTrue(all(rsi.between(0, 100)))
        
        # Validate Stochastic bounds
        stoch_k = features['STOCH_K'].dropna()
        stoch_d = features['STOCH_D'].dropna()
        self.assertTrue(all(stoch_k.between(0, 100)))
        self.assertTrue(all(stoch_d.between(0, 100)))
        
        # Validate Williams %R bounds
        willr = features['WILLR'].dropna()
        self.assertTrue(all(willr.between(-100, 0)))
        
        # Check MACD components (after warmup period)
        warmup = 33  # Default MACD uses 12,26,9
        for col in ['MACD', 'MACD_SIGNAL', 'MACD_HIST']:
            values = features[col].iloc[warmup:]
            self.assertTrue(all(np.isfinite(values)))
    
    def test_trend_features(self):
        """Test trend feature extraction."""
        features = self.extractor._extract_trend_features(self.data)
        
        # Check all trend features are present
        expected_features = {
            'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_10', 'EMA_20', 'EMA_50',
            'ADX', 'CCI', 'DX', 'MINUS_DI', 'PLUS_DI'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate moving averages (after warmup period)
        warmup = 50  # Longest MA period
        for ma in ['SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50']:
            values = features[ma].iloc[warmup:]
            self.assertTrue(all(np.isfinite(values)))
            self.assertTrue(all(values > 0))
        
        # Validate ADX bounds (after warmup)
        adx = features['ADX'].iloc[warmup:]
        self.assertTrue(all(adx.between(0, 100)))
        
        # Validate DI bounds (after warmup)
        minus_di = features['MINUS_DI'].iloc[warmup:]
        plus_di = features['PLUS_DI'].iloc[warmup:]
        self.assertTrue(all(minus_di.between(0, 100)))
        self.assertTrue(all(plus_di.between(0, 100)))
    
    def test_volatility_features(self):
        """Test volatility feature extraction."""
        features = self.extractor._extract_volatility_features(self.data)
        
        # Check all volatility features are present
        expected_features = {
            'ATR', 'NATR', 'TRANGE',
            'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate ATR and TRANGE are positive (after warmup)
        warmup = 14  # ATR period
        atr = features['ATR'].iloc[warmup:]
        trange = features['TRANGE'].dropna()
        self.assertTrue(all(atr >= 0))
        self.assertTrue(all(trange >= 0))
        
        # Validate NATR bounds
        natr = features['NATR'].iloc[warmup:]
        self.assertTrue(all(natr.between(0, 100)))
        
        # Validate Bollinger Bands relationship (after warmup)
        warmup = 20  # BB period
        upper = features['BBANDS_UPPER'].iloc[warmup:]
        middle = features['BBANDS_MIDDLE'].iloc[warmup:]
        lower = features['BBANDS_LOWER'].iloc[warmup:]
        self.assertTrue(all(upper > middle))
        self.assertTrue(all(middle > lower))
    
    def test_volume_features(self):
        """Test volume feature extraction."""
        features = self.extractor._extract_volume_features(self.data)
        
        # Check all volume features are present
        expected_features = {'OBV', 'AD', 'ADOSC'}
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate OBV is finite (after first point)
        obv = features['OBV'].iloc[1:]
        self.assertTrue(all(np.isfinite(obv)))
        
        # Validate A/D Line is finite
        ad = features['AD'].dropna()
        self.assertTrue(all(np.isfinite(ad)))
        
        # Validate Chaikin A/D Oscillator is finite (after warmup)
        warmup = 10  # ADOSC uses 3,10 by default
        adosc = features['ADOSC'].iloc[warmup:]
        self.assertTrue(all(np.isfinite(adosc)))
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create data with insufficient length
        short_data = self.data.iloc[:10]
        
        # Test each feature group
        for group in ['momentum', 'trend', 'volatility', 'volume']:
            extractor = TechnicalFeatureExtractor(feature_groups=[group])
            features = extractor.extract_features(short_data)
            
            # Check that some values are NaN due to insufficient data
            self.assertTrue(features.isna().any().any())
            
            # Check that NaN values are handled
            cleaned = extractor._handle_inf_nan(features)
            self.assertFalse(cleaned.isna().any().any())

if __name__ == '__main__':
    unittest.main() 