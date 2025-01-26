"""
Tests for the training script.
"""

import os
import tempfile
import unittest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from src.training.run_training import setup_logging, train_model
from src.training.config import TrainingConfiguration

class TestRunTraining(unittest.TestCase):
    """Test cases for training script."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a small test dataset
        self.data = pd.DataFrame({
            'price': [1.0] * 100,
            'volume': [1000] * 100,
            'direction': [1] * 100,
            'return': [0.01] * 100,
            'volatility': [0.02] * 100,
            'options_features': [[0.1] * 768] * 100,
            'technical_features': [[0.1] * 768] * 100
        })
        self.data_path = os.path.join(self.temp_dir, 'test_data.parquet')
        self.data.to_parquet(self.data_path)
        
        # Create a test configuration
        self.config = TrainingConfiguration(
            data=dict(
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1,
                seq_length=30
            ),
            model=dict(
                hidden_size=768,
                num_layers=2,
                num_heads=8,
                dropout=0.1
            ),
            training=dict(
                batch_size=4,
                learning_rate=1e-4,
                num_epochs=2,
                device='cpu',
                log_dir=os.path.join(self.temp_dir, 'logs'),
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
            ),
            early_stopping=dict(
                patience=2,
                min_delta=0.01
            ),
            monitoring=dict(
                use_wandb=False,
                project_name='test_project'
            )
        )
    
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
    
    @patch('torch.utils.tensorboard.SummaryWriter')
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
        mock_writer.assert_called_once_with(self.config.training.log_dir)
        
    def test_train_model_early_stopping(self):
        """Test early stopping during training."""
        # Create a configuration with very strict early stopping
        config = self.config
        config.early_stopping.patience = 1
        config.early_stopping.min_delta = 0.0
        
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
        config = self.config
        config.monitoring.use_wandb = True
        
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
        config = self.config
        config.training.device = 'cuda'
        
        # Train model
        train_model(config, self.data_path)
        
        # Check if model was moved to GPU
        checkpoint_path = os.path.join(config.training.checkpoint_dir, 'checkpoint_final.pt')
        checkpoint = torch.load(checkpoint_path)
        self.assertEqual(checkpoint['device'], 'cuda')

if __name__ == '__main__':
    unittest.main() 