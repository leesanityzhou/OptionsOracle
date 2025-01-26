"""
Tests for the base feature extractor.
"""

import unittest
import numpy as np
import pandas as pd
import pytest

from .base_extractor import BaseFeatureExtractor

class MockFeatureExtractor(BaseFeatureExtractor):
    """Mock implementation of BaseFeatureExtractor for testing."""
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock implementation that applies simple transformations."""
        # Validate data types
        for col in data.columns:
            if not np.issubdtype(data[col].dtype, np.number):
                raise ValueError(f"Column {col} must be numeric")
        
        # Apply transformations
        features = pd.DataFrame()
        features['feature1'] = data['value1'] * 2
        features['feature2'] = data['value2'].fillna(0)  # Fill NaN with 0
        features['feature3'] = data['value3'].clip(-100, 100)  # Clip to avoid infinities
        
        # Handle any remaining NaN values
        features = features.fillna(0)
        return features
    
    def handle_inf_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite and NaN values."""
        result = data.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
        result[numeric_cols] = result[numeric_cols].fillna(0)
        return result
    
    def filter_features(self, data: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        """Filter features by name."""
        return data[feature_names]
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the extractor."""
        self._is_fitted = True
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        if not self._is_fitted:
            raise ValueError("Extractor must be fitted before transform")
        return self.extract_features(data)

class TestBaseFeatureExtractor:
    """Test cases for BaseFeatureExtractor."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.extractor = MockFeatureExtractor()
        self.test_data = pd.DataFrame({
            'value1': [0, 1, 2, 3, 4],
            'value2': [1, 2, np.nan, 4, 5],  # Include one NaN value
            'value3': [-50, 2, 3, -75, 5]    # Values within reasonable bounds
        })
    
    def test_validate_data(self):
        """Test data validation."""
        # Valid data should pass
        self.extractor.validate_data(self.test_data)
        
        # Invalid data types should raise ValueError
        invalid_data = pd.DataFrame({
            'value1': ['a', 'b', 'c'],  # Non-numeric data
            'value2': [1, 2, 3]
        })
        with pytest.raises(ValueError, match="must be numeric"):
            self.extractor.extract_features(invalid_data)
    
    def test_handle_inf_nan(self):
        """Test handling of infinite and NaN values."""
        data_with_inf = pd.DataFrame({
            'value1': [np.inf, -np.inf, 1, 2],
            'value2': [np.nan, 1, 2, 3],
            'value3': [1, 2, 3, 4]
        })
        cleaned_data = self.extractor.handle_inf_nan(data_with_inf)
        assert not np.any(np.isinf(cleaned_data.values))
        assert not np.any(np.isnan(cleaned_data.values))
    
    def test_feature_names_filtering(self):
        """Test filtering of feature names."""
        features = self.extractor.extract_features(self.test_data)
        filtered_features = self.extractor.filter_features(features, ['feature1', 'feature2'])
        assert list(filtered_features.columns) == ['feature1', 'feature2']
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        # First handle NaN values
        clean_data = self.extractor.handle_inf_nan(self.test_data)
        
        # Then fit and transform
        result = self.extractor.fit_transform(clean_data)
        assert isinstance(result, pd.DataFrame)
        assert not np.any(np.isinf(result.values))
        assert not np.any(np.isnan(result.values))
    
    def test_transform(self):
        """Test transform method."""
        # First fit the extractor with clean data
        clean_data = self.extractor.handle_inf_nan(self.test_data)
        self.extractor.fit_transform(clean_data)
        
        # Then transform new data
        new_data = pd.DataFrame({
            'value1': [5, 6, 7],
            'value2': [6, 7, 8],
            'value3': [-25, 30, 45]
        })
        result = self.extractor.transform(new_data)
        assert isinstance(result, pd.DataFrame)
        assert not np.any(np.isinf(result.values))
        assert not np.any(np.isnan(result.values))

if __name__ == '__main__':
    unittest.main() 