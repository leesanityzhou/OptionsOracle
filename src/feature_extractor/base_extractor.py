"""
Base feature extractor class providing common functionality for all feature extractors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            feature_names: List of features to extract. If None, extract all available features.
        """
        self.feature_names = feature_names
        self._scaler = StandardScaler()
        self._is_fitted = False
    
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the input data.
        
        Args:
            data: Input DataFrame containing raw data
            
        Returns:
            DataFrame containing extracted features
        """
        pass
    
    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validate that the input data contains required columns.
        
        Args:
            data: Input DataFrame to validate
            required_columns: List of required column names
            
        Raises:
            ValueError: If any required columns are missing
        """
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler and transform the data.
        
        Args:
            data: Input DataFrame to transform
            
        Returns:
            Scaled DataFrame
        """
        # Filter features if specified
        if self.feature_names:
            data = data[self.feature_names]
        
        if not self._is_fitted:
            self._scaler.fit(data)
            self._is_fitted = True
        
        scaled_data = pd.DataFrame(
            self._scaler.transform(data),
            index=data.index,
            columns=data.columns
        )
        
        return scaled_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler.
        
        Args:
            data: Input DataFrame to transform
            
        Returns:
            Scaled DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        # Filter features if specified
        if self.feature_names:
            data = data[self.feature_names]
        
        return pd.DataFrame(
            self._scaler.transform(data),
            index=data.index,
            columns=data.columns
        )
    
    def _handle_inf_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle infinite and NaN values in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with inf and NaN values handled
        """
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df = df.ffill()
        
        # If any NaN values remain at the start, backward fill
        df = df.bfill()
        
        # If any NaN values still remain, fill with 0
        df = df.fillna(0)
        
        # Filter features if specified
        if self.feature_names:
            df = df[self.feature_names]
        
        return df 