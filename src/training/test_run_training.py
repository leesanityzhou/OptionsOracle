"""
Tests for the training script.
"""

import os
import tempfile
import unittest
import pandas as pd
import torch
import yaml
from unittest.mock import patch, MagicMock
import numpy as np

from src.training.run_training import setup_logging, train_model
from src.training.config import TrainingConfiguration

class TestRunTraining(unittest.TestCase):
    """Test cases for training script."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a small test dataset
        num_samples = 1000  # Increase number of samples
        n_embd = 768  # Match model's hidden size
        
        # Create base features
        self.data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=num_samples),
            'price': np.random.randn(num_samples),  # Random price data
            'volume': np.random.randn(num_samples),  # Random volume data
            'direction': [1] * num_samples,
            'returns': [0.01] * num_samples,
            'volatility': [0.02] * num_samples
        })
        
        # Add technical features
        technical_features = np.random.randn(num_samples, n_embd - 2)  # -2 for price and volume
        for i in range(n_embd - 2):
            self.data[f'technical_feature_{i}'] = technical_features[:, i]
        
        self.data_path = os.path.join(self.temp_dir, 'test_data.parquet')
        self.data.to_parquet(self.data_path)
        
        # Create a test configuration file
        self.config_dict = {
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1,
                'seq_length': 30
            },
            'model': {
                'hidden_size': 768,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 1e-4,
                'num_epochs': 2,
                'device': 'cpu',
                'log_dir': os.path.join(self.temp_dir, 'logs'),
                'checkpoint_dir': os.path.join(self.temp_dir, 'checkpoints')
            },
            'early_stopping': {
                'patience': 2,
                'min_delta': 0.01
            },
            'monitoring': {
                'use_wandb': False,
                'project_name': 'test_project'
            }
        }
        
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config_dict, f)
            
        # Load configuration
        self.config = TrainingConfiguration.from_yaml(self.config_path)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_setup_logging(self):
        """Test logging setup."""
        log_dir = os.path.join(self.temp_dir, 'logs')
        setup_logging(log_dir)
        
        # Check if log directory was created
        self.assertTrue(os.path.exists(log_dir))
        
        # Check if log file was created
        log_files = [f for f in os.listdir(log_dir) if f.startswith('training_') and f.endswith('.log')]
        self.assertEqual(len(log_files), 1)
    
    @patch('src.training.run_training.SummaryWriter')
    def test_train_model(self, mock_writer):
        """Test model training process."""
        # Mock tensorboard writer
        mock_writer.return_value = MagicMock()
        
        # Train model
        train_model(self.config, self.data_path)
        
        # Check if checkpoint directory was created
        self.assertTrue(os.path.exists(self.config.training.checkpoint_dir))
        
        # Check if checkpoints were saved
        checkpoints = [f for f in os.listdir(self.config.training.checkpoint_dir) 
                      if f.startswith('checkpoint_') and f.endswith('.pt')]
        self.assertGreater(len(checkpoints), 0)
        
        # Check if tensorboard writer was used
        mock_writer.assert_called_once_with(log_dir=self.config.training.log_dir)
    
    def test_train_model_early_stopping(self):
        """Test early stopping during training."""
        # Create a configuration with very strict early stopping
        config_dict = self.config_dict.copy()
        config_dict['early_stopping'] = {
            'patience': 1,
            'min_delta': 0.0
        }
        
        config_path = os.path.join(self.temp_dir, 'early_stopping_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)
            
        config = TrainingConfiguration.from_yaml(config_path)
        
        # Train model
        train_model(config, self.data_path)
        
        # Check if training stopped early
        checkpoints = [f for f in os.listdir(config.training.checkpoint_dir) 
                      if f.startswith('checkpoint_') and f.endswith('.pt')]
        self.assertLess(len(checkpoints), config.training.num_epochs)
    
    @patch('wandb.init')
    @patch('wandb.log')
    @patch('wandb.finish')
    def test_train_model_with_wandb(self, mock_finish, mock_log, mock_init):
        """Test training with wandb monitoring."""
        # Enable wandb
        config_dict = self.config_dict.copy()
        config_dict['monitoring'] = {
            'use_wandb': True,
            'project_name': 'test_project'
        }
        
        config_path = os.path.join(self.temp_dir, 'wandb_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)
            
        config = TrainingConfiguration.from_yaml(config_path)
        
        # Train model
        train_model(config, self.data_path)
        
        # Check if wandb was used
        mock_init.assert_called_once()
        self.assertGreater(mock_log.call_count, 0)
        mock_finish.assert_called_once()
    
    def test_train_model_gpu(self):
        """Test training on GPU if available."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
        
        # Set device to cuda
        config_dict = self.config_dict.copy()
        config_dict['training']['device'] = 'cuda'
        
        config_path = os.path.join(self.temp_dir, 'gpu_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)
            
        config = TrainingConfiguration.from_yaml(config_path)
        
        # Train model
        train_model(config, self.data_path)
        
        # Check if model was moved to GPU
        checkpoint_path = os.path.join(config.training.checkpoint_dir, 'checkpoint_final.pt')
        checkpoint = torch.load(checkpoint_path)
        self.assertEqual(checkpoint['device'], 'cuda')

if __name__ == '__main__':
    unittest.main() 