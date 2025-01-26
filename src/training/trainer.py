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

from .model import OptionsGPT

class OptionsDataset(Dataset):
    """Dataset for options trading data."""
    
    def __init__(
        self,
        features: torch.Tensor,
        direction_labels: torch.Tensor,
        return_labels: torch.Tensor,
        volatility_labels: torch.Tensor,
        seq_length: int = 30
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature tensor of shape (num_samples, num_features)
            direction_labels: Binary labels for direction prediction
            return_labels: Continuous labels for return prediction
            volatility_labels: Continuous labels for volatility prediction
            seq_length: Length of sequences to create
        """
        self.features = features
        self.direction_labels = direction_labels
        self.return_labels = return_labels
        self.volatility_labels = volatility_labels
        self.seq_length = seq_length
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.features) - self.seq_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (features, labels)
        """
        # Get sequence
        feature_seq = self.features[idx:idx + self.seq_length]
        
        # Get labels (use last timestep's labels)
        direction = self.direction_labels[idx + self.seq_length - 1]
        returns = self.return_labels[idx + self.seq_length - 1]
        volatility = self.volatility_labels[idx + self.seq_length - 1]
        
        # Combine labels
        labels = torch.tensor([direction, returns, volatility])
        
        return feature_seq, labels

class OptionsTrainer:
    """Trainer for options trading model."""
    
    def __init__(
        self,
        model: OptionsGPT,
        train_dataset: OptionsDataset,
        val_dataset: OptionsDataset,
        test_dataset: Optional[OptionsDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            log_interval: Number of steps between logging
        """
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * num_epochs
        
        # Initialize learning rate scheduler with fixed pct_start
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,  # Fixed percentage for warm-up phase
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize best validation loss
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        direction_preds = []
        direction_labels = []
        return_preds = []
        return_labels = []
        volatility_preds = []
        volatility_labels = []
        
        for step, (features, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs_embeds=features, labels=labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            
            # Get predictions
            direction_logits = outputs['direction_logits']
            direction_pred = torch.argmax(direction_logits, dim=-1)
            return_pred = outputs['return_pred'].squeeze()
            volatility_pred = outputs['volatility_pred'].squeeze()
            
            # Store predictions and labels
            direction_preds.extend(direction_pred.detach().cpu().numpy())
            direction_labels.extend(labels[:, 0].long().detach().cpu().numpy())
            return_preds.extend(return_pred.detach().cpu().numpy())
            return_labels.extend(labels[:, 1].detach().cpu().numpy())
            volatility_preds.extend(volatility_pred.detach().cpu().numpy())
            volatility_labels.extend(labels[:, 2].detach().cpu().numpy())
            
            # Log progress
            if (step + 1) % self.log_interval == 0:
                self.logger.info(
                    f"Step {step + 1}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'direction_accuracy': accuracy_score(direction_labels, direction_preds),
            'direction_f1': f1_score(direction_labels, direction_preds, average='weighted'),
            'direction_precision': precision_score(direction_labels, direction_preds, average='weighted'),
            'direction_recall': recall_score(direction_labels, direction_preds, average='weighted'),
            'return_mae': mean_absolute_error(return_labels, return_preds),
            'return_rmse': np.sqrt(mean_squared_error(return_labels, return_preds)),
            'volatility_mae': mean_absolute_error(volatility_labels, volatility_preds),
            'volatility_rmse': np.sqrt(mean_squared_error(volatility_labels, volatility_preds))
        }
        
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        direction_preds = []
        direction_labels = []
        return_preds = []
        return_labels = []
        volatility_preds = []
        volatility_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc="Evaluating"):
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs_embeds=features, labels=labels)
                loss = outputs['loss']
                
                # Update metrics
                total_loss += loss.item()
                
                # Get predictions
                direction_logits = outputs['direction_logits']
                direction_pred = torch.argmax(direction_logits, dim=-1)
                return_pred = outputs['return_pred'].squeeze()
                volatility_pred = outputs['volatility_pred'].squeeze()
                
                # Store predictions and labels
                direction_preds.extend(direction_pred.cpu().numpy())
                direction_labels.extend(labels[:, 0].long().cpu().numpy())
                return_preds.extend(return_pred.cpu().numpy())
                return_labels.extend(labels[:, 1].cpu().numpy())
                volatility_preds.extend(volatility_pred.cpu().numpy())
                volatility_labels.extend(labels[:, 2].cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(dataloader),
            'direction_accuracy': accuracy_score(direction_labels, direction_preds),
            'direction_f1': f1_score(direction_labels, direction_preds, average='weighted'),
            'direction_precision': precision_score(direction_labels, direction_preds, average='weighted'),
            'direction_recall': recall_score(direction_labels, direction_preds, average='weighted'),
            'return_mae': mean_absolute_error(return_labels, return_preds),
            'return_rmse': np.sqrt(mean_squared_error(return_labels, return_preds)),
            'volatility_mae': mean_absolute_error(volatility_labels, volatility_preds),
            'volatility_rmse': np.sqrt(mean_squared_error(volatility_labels, volatility_preds))
        }
        
        return metrics
    
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
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info("Training metrics:")
            for metric, value in train_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            self.logger.info("Validation metrics:")
            for metric, value in val_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, epoch + 1)
        
        self.logger.info("Training completed!")
        
        # Final evaluation on test set
        if self.test_loader is not None:
            self.logger.info("Evaluating on test set...")
            test_metrics = self.evaluate(self.test_loader)
            self.logger.info("Test metrics:")
            for metric, value in test_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}") 