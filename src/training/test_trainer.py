"""
Tests for the training pipeline.
"""

import unittest
import tempfile
import os
import torch
from transformers import GPT2Config

from src.training.model import OptionsGPT
from src.training.trainer import OptionsTrainer, OptionsDataset

class TestOptionsDataset(unittest.TestCase):
    """Test cases for OptionsDataset."""

    def setUp(self):
        """Set up test environment."""
        self.batch_size = 4
        self.seq_length = 5
        self.num_features = 768
        self.num_samples = 100

        # Create dummy data
        self.features = torch.randn(self.num_samples, self.num_features)
        self.direction_labels = torch.randint(0, 2, (self.num_samples,)).float()
        self.return_labels = torch.randn(self.num_samples)
        self.volatility_labels = torch.abs(torch.randn(self.num_samples))

        # Create dataset
        self.dataset = OptionsDataset(
            self.features,
            self.direction_labels,
            self.return_labels,
            self.volatility_labels,
            self.seq_length
        )

    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), self.num_samples - self.seq_length + 1)

    def test_get_item(self):
        """Test getting an item from the dataset."""
        idx = 0
        features, labels = self.dataset[idx]

        self.assertEqual(features.shape, (self.seq_length, self.num_features))
        self.assertEqual(labels.shape, (3,))  # [direction, return, volatility]

class TestOptionsTrainer(unittest.TestCase):
    """Test cases for OptionsTrainer."""

    def setUp(self):
        """Set up test environment."""
        self.batch_size = 4
        self.seq_length = 5  # Reduced sequence length for testing
        self.num_features = 768
        self.num_samples = 100

        # Create dummy data with balanced classes
        features = torch.randn(self.num_samples, self.num_features)
        # Ensure balanced classes (50/50 split)
        direction_labels = torch.cat([
            torch.zeros(self.num_samples // 2),
            torch.ones(self.num_samples // 2)
        ]).float()
        return_labels = torch.randn(self.num_samples)
        volatility_labels = torch.abs(torch.randn(self.num_samples))

        # Create datasets with proper splits
        train_end = 70  # 70% for training
        val_end = 85   # 15% for validation

        self.train_dataset = OptionsDataset(
            features[:train_end],
            direction_labels[:train_end],
            return_labels[:train_end],
            volatility_labels[:train_end],
            self.seq_length
        )

        self.val_dataset = OptionsDataset(
            features[train_end:val_end],
            direction_labels[train_end:val_end],
            return_labels[train_end:val_end],
            volatility_labels[train_end:val_end],
            self.seq_length
        )

        self.test_dataset = OptionsDataset(
            features[val_end:],
            direction_labels[val_end:],
            return_labels[val_end:],
            volatility_labels[val_end:],
            self.seq_length
        )

        # Create model
        config = GPT2Config(
            n_embd=self.num_features,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.1
        )
        self.model = OptionsGPT(config)

        # Create trainer
        self.trainer = OptionsTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_epochs=1,
            device="cpu"
        )

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
        self.assertIsNotNone(self.trainer.train_loader)
        self.assertIsNotNone(self.trainer.val_loader)
        self.assertIsNotNone(self.trainer.test_loader)

    def test_train_epoch(self):
        """Test training for one epoch."""
        metrics = self.trainer.train_epoch()
        self.assertIn('loss', metrics)
        self.assertIn('direction_accuracy', metrics)
        self.assertIn('return_mae', metrics)
        self.assertIn('return_rmse', metrics)
        self.assertIn('volatility_mae', metrics)
        self.assertIn('volatility_rmse', metrics)

    def test_evaluate(self):
        """Test model evaluation."""
        metrics = self.trainer.evaluate(self.trainer.val_loader)
        self.assertIn('loss', metrics)
        self.assertIn('direction_accuracy', metrics)
        self.assertIn('return_mae', metrics)
        self.assertIn('return_rmse', metrics)
        self.assertIn('volatility_mae', metrics)
        self.assertIn('volatility_rmse', metrics)

    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer.checkpoint_dir = temp_dir

            # Train for one epoch
            metrics = self.trainer.train_epoch()

            # Save checkpoint
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pt')
            self.trainer.save_checkpoint(metrics, epoch=1)

            # Check if checkpoint files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "checkpoint_epoch_1.pt")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "best_model.pt")))

    def test_full_training(self):
        """Test full training loop."""
        # Train model
        self.trainer.train()

        # Evaluate on test set
        test_metrics = self.trainer.evaluate(self.trainer.test_loader)
        self.assertIn('loss', test_metrics)
        self.assertIn('direction_accuracy', test_metrics)
        self.assertIn('return_mae', test_metrics)
        self.assertIn('return_rmse', test_metrics)
        self.assertIn('volatility_mae', test_metrics)
        self.assertIn('volatility_rmse', test_metrics)

if __name__ == '__main__':
    unittest.main() 