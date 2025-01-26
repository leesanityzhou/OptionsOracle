"""
Options data feature extractor for options chain analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base_extractor import BaseFeatureExtractor
from ..utils.config import get_feature_groups

logger = logging.getLogger(__name__)

class OptionsFeatureExtractor(BaseFeatureExtractor):
    """Extract features from options chain data."""
    
    def __init__(
        self,
        feature_groups: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        atm_threshold: float = 0.05
    ):
        """
        Initialize the options feature extractor.
        
        Args:
            feature_groups: List of feature groups to extract
            feature_names: List of specific features to extract
            atm_threshold: Threshold for considering an option at-the-money (as % of stock price)
        """
        super().__init__(feature_names)
        
        # Load feature groups from config
        self.FEATURE_GROUPS = get_feature_groups('options')
        
        # Set feature groups to extract
        self.feature_groups = feature_groups or list(self.FEATURE_GROUPS.keys())
        self.atm_threshold = atm_threshold
        
        # Validate feature groups
        invalid_groups = [g for g in self.feature_groups if g not in self.FEATURE_GROUPS]
        if invalid_groups:
            raise ValueError(f"Invalid feature groups: {invalid_groups}")
    
    def extract_features(
        self,
        calls_data: pd.DataFrame,
        puts_data: pd.DataFrame,
        stock_price: float
    ) -> pd.DataFrame:
        """
        Extract features from options chain data.
        
        Args:
            calls_data: DataFrame with call options data
            puts_data: DataFrame with put options data
            stock_price: Current stock price
            
        Returns:
            DataFrame with options features
        """
        try:
            # Get ATM options
            atm_calls = self._get_atm_options(calls_data, stock_price)
            atm_puts = self._get_atm_options(puts_data, stock_price)
            
            # Initialize features DataFrame with one row
            features = pd.DataFrame(index=[0])
            
            # Extract features by group
            if 'pricing' in self.feature_groups:
                pricing_features = self._extract_pricing_features(atm_calls, atm_puts)
                features = pd.concat([features, pricing_features], axis=1)
            
            if 'volume' in self.feature_groups:
                volume_features = self._extract_volume_features(
                    calls_data, puts_data,
                    atm_calls, atm_puts
                )
                features = pd.concat([features, volume_features], axis=1)
            
            if 'open_interest' in self.feature_groups:
                oi_features = self._extract_open_interest_features(
                    calls_data, puts_data,
                    atm_calls, atm_puts
                )
                features = pd.concat([features, oi_features], axis=1)
            
            if 'implied_volatility' in self.feature_groups:
                iv_features = self._extract_iv_features(
                    calls_data, puts_data,
                    atm_calls, atm_puts,
                    stock_price
                )
                features = pd.concat([features, iv_features], axis=1)
            
            if 'greeks' in self.feature_groups:
                greeks_features = self._extract_greeks_features(
                    calls_data, puts_data,
                    atm_calls, atm_puts,
                    stock_price
                )
                features = pd.concat([features, greeks_features], axis=1)
            
            if 'risk_metrics' in self.feature_groups:
                risk_features = self._extract_risk_metrics(
                    calls_data, puts_data,
                    atm_calls, atm_puts,
                    stock_price,
                    features
                )
                features = pd.concat([features, risk_features], axis=1)
            
            # Handle inf and nan values
            features = self._handle_inf_nan(features)
            
            # Filter features if specific ones are requested
            if self.feature_names:
                features = features[self.feature_names]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting options features: {e}")
            raise
    
    def _get_atm_options(
        self,
        df: pd.DataFrame,
        stock_price: float
    ) -> pd.DataFrame:
        """Get at-the-money options."""
        # Ensure required columns exist
        required_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate moneyness
        moneyness = abs(1 - df['strike'] / stock_price)
        
        # Get ATM options
        atm_options = df[moneyness <= self.atm_threshold].copy()
        
        # If no options are within threshold, get the closest one
        if atm_options.empty:
            closest_idx = moneyness.idxmin()
            atm_options = df.loc[[closest_idx]].copy()
        
        return atm_options
    
    def _extract_pricing_features(
        self,
        atm_calls: pd.DataFrame,
        atm_puts: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract pricing features."""
        df = pd.DataFrame(index=[0])  # Initialize with one row
        
        # ATM prices
        df['ATM_CALL_PRICE'] = atm_calls['lastPrice'].mean()
        df['ATM_PUT_PRICE'] = atm_puts['lastPrice'].mean()
        
        # ATM spreads
        df['ATM_CALL_ASK_BID_SPREAD'] = (
            atm_calls['ask'] - atm_calls['bid']
        ).mean()
        df['ATM_PUT_ASK_BID_SPREAD'] = (
            atm_puts['ask'] - atm_puts['bid']
        ).mean()
        
        return df
    
    def _extract_volume_features(
        self,
        calls_data: pd.DataFrame,
        puts_data: pd.DataFrame,
        atm_calls: pd.DataFrame,
        atm_puts: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract volume features."""
        df = pd.DataFrame(index=[0])  # Initialize with one row
        
        # Total volume
        df['TOTAL_CALL_VOLUME'] = calls_data['volume'].sum()
        df['TOTAL_PUT_VOLUME'] = puts_data['volume'].sum()
        
        # Put/Call volume ratio
        total_call_volume = df['TOTAL_CALL_VOLUME'].iloc[0]
        total_put_volume = df['TOTAL_PUT_VOLUME'].iloc[0]
        df['PUT_CALL_VOLUME_RATIO'] = (
            total_put_volume / total_call_volume
            if total_call_volume > 0 else 0
        )
        
        # ATM volume
        df['ATM_CALL_VOLUME'] = atm_calls['volume'].sum()
        df['ATM_PUT_VOLUME'] = atm_puts['volume'].sum()
        
        return df
    
    def _extract_open_interest_features(
        self,
        calls_data: pd.DataFrame,
        puts_data: pd.DataFrame,
        atm_calls: pd.DataFrame,
        atm_puts: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract open interest features."""
        df = pd.DataFrame(index=[0])  # Initialize with one row
        
        # Total OI
        df['TOTAL_CALL_OI'] = calls_data['openInterest'].sum()
        df['TOTAL_PUT_OI'] = puts_data['openInterest'].sum()
        
        # Put/Call OI ratio
        total_call_oi = df['TOTAL_CALL_OI'].iloc[0]
        total_put_oi = df['TOTAL_PUT_OI'].iloc[0]
        df['PUT_CALL_OI_RATIO'] = (
            total_put_oi / total_call_oi
            if total_call_oi > 0 else 0
        )
        
        # ATM OI
        df['ATM_CALL_OI'] = atm_calls['openInterest'].sum()
        df['ATM_PUT_OI'] = atm_puts['openInterest'].sum()
        
        return df
    
    def _extract_iv_features(
        self,
        calls_data: pd.DataFrame,
        puts_data: pd.DataFrame,
        atm_calls: pd.DataFrame,
        atm_puts: pd.DataFrame,
        stock_price: float
    ) -> pd.DataFrame:
        """Extract implied volatility features."""
        df = pd.DataFrame(index=[0])  # Initialize with one row
        
        # ATM IV
        df['ATM_CALL_IV'] = atm_calls['impliedVolatility'].mean()
        df['ATM_PUT_IV'] = atm_puts['impliedVolatility'].mean()
        
        # Volume-weighted IV
        total_call_volume = calls_data['volume'].sum()
        total_put_volume = puts_data['volume'].sum()
        
        df['WEIGHTED_CALL_IV'] = (
            (calls_data['impliedVolatility'] * calls_data['volume']).sum() /
            total_call_volume if total_call_volume > 0 else 0
        )
        
        df['WEIGHTED_PUT_IV'] = (
            (puts_data['impliedVolatility'] * puts_data['volume']).sum() /
            total_put_volume if total_put_volume > 0 else 0
        )
        
        # IV Skew (OTM puts vs ATM puts)
        otm_puts = puts_data[puts_data['strike'] < stock_price * 0.9]
        if not otm_puts.empty:
            df['IV_SKEW'] = (
                otm_puts['impliedVolatility'].mean() -
                df['ATM_PUT_IV'].iloc[0]
            )
        else:
            df['IV_SKEW'] = 0
        
        # IV Term Structure (not implemented as we need multiple expiries)
        df['IV_TERM_STRUCTURE'] = 0
        
        return df
    
    def _extract_greeks_features(
        self,
        calls_data: pd.DataFrame,
        puts_data: pd.DataFrame,
        atm_calls: pd.DataFrame,
        atm_puts: pd.DataFrame,
        stock_price: float
    ) -> pd.DataFrame:
        """Extract Greeks-related features."""
        df = pd.DataFrame(index=[0])  # Initialize with one row
        
        # Delta features
        df['ATM_CALL_DELTA'] = atm_calls['delta'].mean()
        df['ATM_PUT_DELTA'] = atm_puts['delta'].mean()
        
        # Delta skew
        otm_calls = calls_data[calls_data['strike'] > stock_price * 1.1]
        otm_puts = puts_data[puts_data['strike'] < stock_price * 0.9]
        
        df['DELTA_SKEW_CALLS'] = (
            otm_calls['delta'].mean() - df['ATM_CALL_DELTA'].iloc[0]
            if not otm_calls.empty else 0
        )
        
        df['DELTA_SKEW_PUTS'] = (
            otm_puts['delta'].mean() - df['ATM_PUT_DELTA'].iloc[0]
            if not otm_puts.empty else 0
        )
        
        # Volume-weighted delta
        total_call_volume = calls_data['volume'].sum()
        total_put_volume = puts_data['volume'].sum()
        
        df['WEIGHTED_CALL_DELTA'] = (
            (calls_data['delta'] * calls_data['volume']).sum() /
            total_call_volume if total_call_volume > 0 else 0
        )
        
        df['WEIGHTED_PUT_DELTA'] = (
            (puts_data['delta'] * puts_data['volume']).sum() /
            total_put_volume if total_put_volume > 0 else 0
        )
        
        # Gamma features
        df['ATM_CALL_GAMMA'] = atm_calls['gamma'].mean()
        df['ATM_PUT_GAMMA'] = atm_puts['gamma'].mean()
        
        # Gamma exposure (gamma * open interest * stock price)
        df['GAMMA_EXPOSURE_CALLS'] = (
            (calls_data['gamma'] * calls_data['openInterest'] * stock_price).sum()
        )
        df['GAMMA_EXPOSURE_PUTS'] = (
            (puts_data['gamma'] * puts_data['openInterest'] * stock_price).sum()
        )
        df['TOTAL_GAMMA_EXPOSURE'] = (
            df['GAMMA_EXPOSURE_CALLS'].iloc[0] + df['GAMMA_EXPOSURE_PUTS'].iloc[0]
        )
        
        # Theta features
        df['ATM_CALL_THETA'] = atm_calls['theta'].mean()
        df['ATM_PUT_THETA'] = atm_puts['theta'].mean()
        
        # Theta decay ratio (theta / option price)
        df['THETA_DECAY_RATIO_CALLS'] = (
            (atm_calls['theta'] / atm_calls['lastPrice']).mean()
        )
        df['THETA_DECAY_RATIO_PUTS'] = (
            (atm_puts['theta'] / atm_puts['lastPrice']).mean()
        )
        
        # Vega features
        df['ATM_CALL_VEGA'] = atm_calls['vega'].mean()
        df['ATM_PUT_VEGA'] = atm_puts['vega'].mean()
        
        # Volume-weighted vega
        df['VEGA_WEIGHTED_CALLS'] = (
            (calls_data['vega'] * calls_data['volume']).sum() /
            total_call_volume if total_call_volume > 0 else 0
        )
        
        df['VEGA_WEIGHTED_PUTS'] = (
            (puts_data['vega'] * puts_data['volume']).sum() /
            total_put_volume if total_put_volume > 0 else 0
        )
        
        # Rho features
        df['ATM_CALL_RHO'] = atm_calls['rho'].mean()
        df['ATM_PUT_RHO'] = atm_puts['rho'].mean()
        
        return df
    
    def _extract_risk_metrics(
        self,
        calls_data: pd.DataFrame,
        puts_data: pd.DataFrame,
        atm_calls: pd.DataFrame,
        atm_puts: pd.DataFrame,
        stock_price: float,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract risk metrics based on Greeks and other factors."""
        df = pd.DataFrame(index=[0])  # Initialize with one row
        
        # Put-Call parity deviation for ATM options
        # (C - P) - (S - K*e^(-rt))
        # Simplified version without interest rate
        df['PUT_CALL_PARITY_DEVIATION'] = (
            (atm_calls['lastPrice'].mean() - atm_puts['lastPrice'].mean()) -
            (stock_price - atm_calls['strike'].mean())
        )
        
        # Delta neutral ratio (ratio of puts to calls needed for delta neutrality)
        atm_call_delta = features['ATM_CALL_DELTA'].iloc[0]
        atm_put_delta = features['ATM_PUT_DELTA'].iloc[0]
        df['DELTA_NEUTRAL_RATIO'] = abs(
            atm_call_delta / atm_put_delta
            if atm_put_delta != 0 else 0
        )
        
        # Gamma scalping opportunity
        # High gamma and volume indicate better scalping opportunities
        df['GAMMA_SCALPING_OPPORTUNITY'] = (
            features['ATM_CALL_GAMMA'].iloc[0] * atm_calls['volume'].mean() +
            features['ATM_PUT_GAMMA'].iloc[0] * atm_puts['volume'].mean()
        )
        
        # Theta/Vega ratio (higher means more premium decay relative to volatility exposure)
        total_theta = features['ATM_CALL_THETA'].iloc[0] + features['ATM_PUT_THETA'].iloc[0]
        total_vega = features['ATM_CALL_VEGA'].iloc[0] + features['ATM_PUT_VEGA'].iloc[0]
        df['THETA_VEGA_RATIO'] = (
            total_theta / total_vega
            if total_vega != 0 else 0
        )
        
        return df
    
    def _handle_inf_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite and NaN values in features."""
        # Replace inf with large values
        df = df.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Replace NaN with 0
        df = df.fillna(0)
        
        return df 