"""
Tests for the training script.
"""

import unittest
import tempfile
import os
import pandas as pd
import torch
from src.training.train import prepare_data
from src.training.trainer import OptionsDataset

class TestTrainingScript(unittest.TestCase):
    """Test cases for training script."""

    def setUp(self):
        """Set up test environment."""
        self.num_samples = 100
        self.num_features = 10
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy data
        data = {
            'date': pd.date_range(start='2023-01-01', periods=self.num_samples),
            'direction': [0, 1] * (self.num_samples // 2),
            'returns': [0.1, -0.1] * (self.num_samples // 2),
            'volatility': [0.2, 0.3] * (self.num_samples // 2)
        }
        
        # Add feature columns
        for i in range(self.num_features):
            data[f'feature_{i}'] = torch.randn(self.num_samples).numpy()
            
        # Create DataFrame
        self.df = pd.DataFrame(data)
        
        # Save to parquet
        self.data_path = os.path.join(self.temp_dir, 'test_data.parquet')
        self.df.to_parquet(self.data_path)

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_prepare_data(self):
        """Test data preparation function."""
        seq_length = 5
        
        # Test with default parameters
        train_dataset, val_dataset, test_dataset = prepare_data(
            self.data_path,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seq_length=seq_length
        )
        
        # Check dataset types
        self.assertIsInstance(train_dataset, OptionsDataset)
        self.assertIsInstance(val_dataset, OptionsDataset)
        self.assertIsInstance(test_dataset, OptionsDataset)
        
        # Calculate split sizes for raw data
        total_raw_samples = len(self.df)
        train_raw_size = int(total_raw_samples * 0.7)
        val_raw_size = int(total_raw_samples * 0.15)
        test_raw_size = total_raw_samples - train_raw_size - val_raw_size
        
        # Calculate expected dataset sizes after sequence creation
        expected_train_size = train_raw_size - seq_length + 1
        expected_val_size = val_raw_size - seq_length + 1
        expected_test_size = test_raw_size - seq_length + 1
        
        # Check dataset sizes
        self.assertEqual(len(train_dataset), expected_train_size)
        self.assertEqual(len(val_dataset), expected_val_size)
        self.assertEqual(len(test_dataset), expected_test_size)
        
        # Test data splitting with different ratios
        train_dataset, val_dataset, test_dataset = prepare_data(
            self.data_path,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seq_length=seq_length
        )
        
        # Calculate split sizes for raw data
        train_raw_size = int(total_raw_samples * 0.8)
        val_raw_size = int(total_raw_samples * 0.1)
        test_raw_size = total_raw_samples - train_raw_size - val_raw_size
        
        # Calculate expected dataset sizes after sequence creation
        expected_train_size = train_raw_size - seq_length + 1
        expected_val_size = val_raw_size - seq_length + 1
        expected_test_size = test_raw_size - seq_length + 1
        
        # Check dataset sizes
        self.assertEqual(len(train_dataset), expected_train_size)
        self.assertEqual(len(val_dataset), expected_val_size)
        self.assertEqual(len(test_dataset), expected_test_size)
        
    def test_prepare_data_invalid_ratios(self):
        """Test data preparation with invalid ratios."""
        # Test with ratios that don't sum to 1
        with self.assertRaises(ValueError):
            prepare_data(
                self.data_path,
                train_ratio=0.8,
                val_ratio=0.3,
                test_ratio=0.2,
                seq_length=5
            )
            
        # Test with negative ratios
        with self.assertRaises(ValueError):
            prepare_data(
                self.data_path,
                train_ratio=-0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                seq_length=5
            )
            
    def test_prepare_data_invalid_sequence_length(self):
        """Test data preparation with invalid sequence length."""
        # Test with sequence length longer than dataset
        with self.assertRaises(ValueError):
            prepare_data(
                self.data_path,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                seq_length=self.num_samples + 1
            )
            
        # Test with negative sequence length
        with self.assertRaises(ValueError):
            prepare_data(
                self.data_path,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                seq_length=-1
            )

    def test_prepare_data_missing_columns(self):
        """Test data preparation with missing required columns."""
        # Create data without required columns
        data = {
            'date': pd.date_range(start='2023-01-01', periods=self.num_samples),
            'feature_1': torch.randn(self.num_samples).numpy()
        }
        df = pd.DataFrame(data)
        
        # Save to parquet
        invalid_data_path = os.path.join(self.temp_dir, 'invalid_data.parquet')
        df.to_parquet(invalid_data_path)
        
        # Test with missing columns
        with self.assertRaises(ValueError):
            prepare_data(
                invalid_data_path,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                seq_length=5
            )

if __name__ == '__main__':
    unittest.main() 