"""
Training pipeline for options trading model.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
import numpy as np
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F

from .model import OptionsGPT

class OptionsDataset(Dataset):
    """Dataset for stock price prediction with time-based weighting."""
    
    def __init__(
        self,
        features: torch.Tensor,
        price_labels: torch.Tensor,
        seq_length: int = 10  # Changed to 10 days (2 weeks)
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature tensor of shape (num_samples, num_features)
            price_labels: Next day price labels
            seq_length: Length of sequences to create (default: 10 days)
        """
        self.features = features
        self.price_labels = price_labels
        self.seq_length = seq_length
        
        # Create time-based weights (more recent days have higher weights)
        self.time_weights = torch.exp(torch.linspace(0, 1, seq_length))  # Exponential weighting
        self.time_weights = self.time_weights / self.time_weights.sum()  # Normalize weights
        
    def __len__(self):
        # Return the number of possible sequences we can create
        # For a sequence length of 10, we need at least 10 days of data
        if len(self.features) < self.seq_length:
            return 0
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (features, label, weights)
        """
        # Get sequence
        feature_seq = self.features[idx:idx + self.seq_length]
        
        # Get label (next day's price)
        price = self.price_labels[idx + self.seq_length - 1]
        
        return feature_seq, price, self.time_weights

class OptionsTrainer:
    """Trainer for stock price prediction."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: OptionsDataset,
        val_dataset: OptionsDataset,
        test_dataset: Optional[OptionsDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100
    ) -> None:
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.best_val_loss = float('inf')
        
        # Ensure batch size is at least 1 but no larger than dataset size
        train_batch_size = max(1, min(batch_size, len(train_dataset)))
        val_batch_size = max(1, min(batch_size, len(val_dataset)))
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=max(1, min(batch_size, len(test_dataset))),
                shuffle=False,
                num_workers=0
            )
        else:
            self.test_loader = None
        
        # Set up optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,  # L2 regularization
            betas=(0.9, 0.999),  # Default AdamW betas
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        num_training_steps = len(self.train_loader) * num_epochs
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.max_grad_norm = 1.0
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            features, labels, time_weights = [x.to(self.device) for x in batch]
            
            # Forward pass
            outputs = self.model(features, time_weights)
            price_pred = outputs['price_pred']
            
            # Calculate loss
            loss = F.mse_loss(price_pred, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                features, labels, time_weights = [x.to(self.device) for x in batch]
                
                # Forward pass
                outputs = self.model(features, time_weights)
                price_pred = outputs['price_pred']
                
                # Calculate loss
                loss = F.mse_loss(price_pred, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.extend(price_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        mae = np.mean(np.abs(all_preds - all_labels))
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        
        return {
            'loss': total_loss / num_batches,
            'mae': mae,
            'rmse': rmse
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], epoch: int):
        """
        Save model checkpoint.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model, save it separately
        if metrics['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['loss']
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"New best model saved with validation loss: {metrics['loss']:.4f}")
    
    def train(self):
        """Train model."""
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train epoch
            train_loss = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics['loss']
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"Val RMSE: {val_metrics['rmse']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(val_metrics, epoch + 1)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        self.logger.info("\nTraining completed!")
        
        # Final evaluation on test set
        if self.test_loader is not None:
            self.logger.info("\nEvaluating on test set...")
            test_metrics = self.evaluate(self.test_loader)
            self.logger.info("\nTest metrics:")
            for metric, value in test_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
                
        self.logger.info("\nFinal best validation loss: {:.4f}".format(self.best_val_loss))
        self.logger.info("Best model saved to: {}".format(os.path.join(self.checkpoint_dir, "best_model.pt"))) 