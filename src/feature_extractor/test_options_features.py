"""
Tests for options feature extraction methods.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.feature_extractor import OptionsFeatureExtractor

class TestOptionsFeatures(unittest.TestCase):
    """Test cases for options feature extraction methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = OptionsFeatureExtractor()
        
        # Create sample options data
        np.random.seed(42)
        self.stock_price = 100.0
        
        # Generate call options data
        strikes = np.linspace(80, 120, 20)  # Range of strikes around stock price
        self.calls_data = pd.DataFrame({
            'strike': strikes,
            'lastPrice': [max(0, self.stock_price - k) for k in strikes],
            'bid': [max(0, self.stock_price - k - 0.5) for k in strikes],
            'ask': [max(0, self.stock_price - k + 0.5) for k in strikes],
            'volume': np.random.randint(0, 1000, len(strikes)),
            'openInterest': np.random.randint(100, 5000, len(strikes)),
            'impliedVolatility': np.random.uniform(0.2, 0.5, len(strikes)),
            'delta': [max(0, 1 - k/self.stock_price) for k in strikes],
            'gamma': np.random.uniform(0, 0.1, len(strikes)),
            'theta': -np.random.uniform(0, 0.1, len(strikes)),
            'vega': np.random.uniform(0, 0.2, len(strikes)),
            'rho': np.random.uniform(0, 0.1, len(strikes)),
            'daysToExpiry': [30] * len(strikes)
        })
        
        # Generate put options data
        self.puts_data = pd.DataFrame({
            'strike': strikes,
            'lastPrice': [max(0, k - self.stock_price) for k in strikes],
            'bid': [max(0, k - self.stock_price - 0.5) for k in strikes],
            'ask': [max(0, k - self.stock_price + 0.5) for k in strikes],
            'volume': np.random.randint(0, 1000, len(strikes)),
            'openInterest': np.random.randint(100, 5000, len(strikes)),
            'impliedVolatility': np.random.uniform(0.2, 0.5, len(strikes)),
            'delta': [min(0, -1 + k/self.stock_price) for k in strikes],
            'gamma': np.random.uniform(0, 0.1, len(strikes)),
            'theta': -np.random.uniform(0, 0.1, len(strikes)),
            'vega': np.random.uniform(0, 0.2, len(strikes)),
            'rho': -np.random.uniform(0, 0.1, len(strikes)),
            'daysToExpiry': [30] * len(strikes)
        })
    
    def test_get_atm_options(self):
        """Test ATM options selection."""
        # Test with default threshold
        atm_calls = self.extractor._get_atm_options(self.calls_data, self.stock_price)
        atm_puts = self.extractor._get_atm_options(self.puts_data, self.stock_price)
        
        # Check that ATM options are within threshold
        self.assertTrue(all(
            abs(1 - strike/self.stock_price) <= self.extractor.atm_threshold
            for strike in atm_calls['strike']
        ))
        self.assertTrue(all(
            abs(1 - strike/self.stock_price) <= self.extractor.atm_threshold
            for strike in atm_puts['strike']
        ))
        
        # Test with custom threshold
        narrow_extractor = OptionsFeatureExtractor(atm_threshold=0.01)
        narrow_calls = narrow_extractor._get_atm_options(self.calls_data, self.stock_price)
        
        # With narrow threshold, we should get at least one option (the closest to ATM)
        self.assertEqual(len(narrow_calls), 1)
        closest_strike = narrow_calls['strike'].iloc[0]
        moneyness = abs(1 - closest_strike/self.stock_price)
        
        # Verify it's the closest option
        all_moneyness = abs(1 - self.calls_data['strike']/self.stock_price)
        self.assertEqual(moneyness, all_moneyness.min())
    
    def test_pricing_features(self):
        """Test pricing feature extraction."""
        atm_calls = self.extractor._get_atm_options(self.calls_data, self.stock_price)
        atm_puts = self.extractor._get_atm_options(self.puts_data, self.stock_price)
        
        features = self.extractor._extract_pricing_features(atm_calls, atm_puts)
        
        # Check all pricing features are present
        expected_features = {
            'ATM_CALL_PRICE', 'ATM_PUT_PRICE',
            'ATM_CALL_ASK_BID_SPREAD', 'ATM_PUT_ASK_BID_SPREAD'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate price relationships
        self.assertTrue(all(features['ATM_CALL_PRICE'] >= 0))
        self.assertTrue(all(features['ATM_PUT_PRICE'] >= 0))
        
        # Validate spreads are non-negative
        self.assertTrue(all(features['ATM_CALL_ASK_BID_SPREAD'] >= 0))
        self.assertTrue(all(features['ATM_PUT_ASK_BID_SPREAD'] >= 0))
    
    def test_volume_features(self):
        """Test volume feature extraction."""
        atm_calls = self.extractor._get_atm_options(self.calls_data, self.stock_price)
        atm_puts = self.extractor._get_atm_options(self.puts_data, self.stock_price)
        
        features = self.extractor._extract_volume_features(
            self.calls_data, self.puts_data,
            atm_calls, atm_puts
        )
        
        # Check all volume features are present
        expected_features = {
            'TOTAL_CALL_VOLUME', 'TOTAL_PUT_VOLUME',
            'PUT_CALL_VOLUME_RATIO',
            'ATM_CALL_VOLUME', 'ATM_PUT_VOLUME'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate volumes are non-negative
        self.assertTrue(all(features['TOTAL_CALL_VOLUME'] >= 0))
        self.assertTrue(all(features['TOTAL_PUT_VOLUME'] >= 0))
        self.assertTrue(all(features['ATM_CALL_VOLUME'] >= 0))
        self.assertTrue(all(features['ATM_PUT_VOLUME'] >= 0))
        
        # Validate ratio calculation
        expected_ratio = (
            features['TOTAL_PUT_VOLUME'] /
            features['TOTAL_CALL_VOLUME']
        ).iloc[0]
        self.assertAlmostEqual(
            features['PUT_CALL_VOLUME_RATIO'].iloc[0],
            expected_ratio
        )
    
    def test_open_interest_features(self):
        """Test open interest feature extraction."""
        atm_calls = self.extractor._get_atm_options(self.calls_data, self.stock_price)
        atm_puts = self.extractor._get_atm_options(self.puts_data, self.stock_price)
        
        features = self.extractor._extract_open_interest_features(
            self.calls_data, self.puts_data,
            atm_calls, atm_puts
        )
        
        # Check all OI features are present
        expected_features = {
            'TOTAL_CALL_OI', 'TOTAL_PUT_OI',
            'PUT_CALL_OI_RATIO',
            'ATM_CALL_OI', 'ATM_PUT_OI'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate OI values are non-negative
        self.assertTrue(all(features['TOTAL_CALL_OI'] >= 0))
        self.assertTrue(all(features['TOTAL_PUT_OI'] >= 0))
        self.assertTrue(all(features['ATM_CALL_OI'] >= 0))
        self.assertTrue(all(features['ATM_PUT_OI'] >= 0))
        
        # Validate ratio calculation
        expected_ratio = (
            features['TOTAL_PUT_OI'] /
            features['TOTAL_CALL_OI']
        ).iloc[0]
        self.assertAlmostEqual(
            features['PUT_CALL_OI_RATIO'].iloc[0],
            expected_ratio
        )
    
    def test_iv_features(self):
        """Test implied volatility feature extraction."""
        atm_calls = self.extractor._get_atm_options(self.calls_data, self.stock_price)
        atm_puts = self.extractor._get_atm_options(self.puts_data, self.stock_price)
        
        features = self.extractor._extract_iv_features(
            self.calls_data, self.puts_data,
            atm_calls, atm_puts,
            self.stock_price
        )
        
        # Check all IV features are present
        expected_features = {
            'ATM_CALL_IV', 'ATM_PUT_IV',
            'WEIGHTED_CALL_IV', 'WEIGHTED_PUT_IV',
            'IV_SKEW', 'IV_TERM_STRUCTURE'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate IV values are positive
        self.assertTrue(all(features['ATM_CALL_IV'] > 0))
        self.assertTrue(all(features['ATM_PUT_IV'] > 0))
        self.assertTrue(all(features['WEIGHTED_CALL_IV'] > 0))
        self.assertTrue(all(features['WEIGHTED_PUT_IV'] > 0))
        
        # Validate IV skew calculation
        otm_puts = self.puts_data[
            self.puts_data['strike'] < self.stock_price * 0.9
        ]
        if not otm_puts.empty:
            expected_skew = (
                otm_puts['impliedVolatility'].mean() -
                features['ATM_PUT_IV'].iloc[0]
            )
            self.assertAlmostEqual(
                features['IV_SKEW'].iloc[0],
                expected_skew
            )
    
    def test_greeks_features(self):
        """Test Greeks feature extraction."""
        atm_calls = self.extractor._get_atm_options(self.calls_data, self.stock_price)
        atm_puts = self.extractor._get_atm_options(self.puts_data, self.stock_price)
        
        features = self.extractor._extract_greeks_features(
            self.calls_data, self.puts_data,
            atm_calls, atm_puts,
            self.stock_price
        )
        
        # Check delta features
        self.assertTrue(all(features['ATM_CALL_DELTA'] >= 0))
        self.assertTrue(all(features['ATM_PUT_DELTA'] <= 0))
        
        # Check gamma features
        self.assertTrue(all(features['ATM_CALL_GAMMA'] >= 0))
        self.assertTrue(all(features['ATM_PUT_GAMMA'] >= 0))
        
        # Check theta features
        self.assertTrue(all(features['ATM_CALL_THETA'] <= 0))
        self.assertTrue(all(features['ATM_PUT_THETA'] <= 0))
        
        # Check vega features
        self.assertTrue(all(features['ATM_CALL_VEGA'] >= 0))
        self.assertTrue(all(features['ATM_PUT_VEGA'] >= 0))
        
        # Check rho features
        self.assertTrue(all(np.isfinite(features['ATM_CALL_RHO'])))
        self.assertTrue(all(np.isfinite(features['ATM_PUT_RHO'])))
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        atm_calls = self.extractor._get_atm_options(self.calls_data, self.stock_price)
        atm_puts = self.extractor._get_atm_options(self.puts_data, self.stock_price)
        
        # First get Greeks features for risk metrics calculation
        greeks = self.extractor._extract_greeks_features(
            self.calls_data, self.puts_data,
            atm_calls, atm_puts,
            self.stock_price
        )
        
        features = self.extractor._extract_risk_metrics(
            self.calls_data, self.puts_data,
            atm_calls, atm_puts,
            self.stock_price,
            greeks
        )
        
        # Check all risk metrics are present
        expected_features = {
            'PUT_CALL_PARITY_DEVIATION',
            'DELTA_NEUTRAL_RATIO',
            'GAMMA_SCALPING_OPPORTUNITY',
            'THETA_VEGA_RATIO'
        }
        self.assertEqual(set(features.columns), expected_features)
        
        # Validate risk metrics
        self.assertTrue(all(features['DELTA_NEUTRAL_RATIO'] >= 0))
        self.assertTrue(all(features['GAMMA_SCALPING_OPPORTUNITY'] >= 0))
        self.assertTrue(all(np.isfinite(features['THETA_VEGA_RATIO'])))

if __name__ == '__main__':
    unittest.main() 