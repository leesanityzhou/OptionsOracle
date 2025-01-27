"""
Train model on recent data for next-day prediction using rolling window validation.
"""

import torch
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

from src.training.model import OptionsGPT, OptionsDataset, OptionsGPTConfig
from src.training.trainer import OptionsTrainer
from transformers import GPT2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_window(model, dataset, device="cpu"):
    """Evaluate model on a single window."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, labels, time_weights in loader:
            features = features.to(device)
            labels = labels.to(device)
            time_weights = time_weights.to(device)
            
            outputs = model(features, time_weights)
            price_pred = outputs['price_pred']
            
            loss = torch.nn.functional.mse_loss(price_pred, labels)
            total_loss += loss.item() * len(labels)
            
            predictions.extend(price_pred.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
    
    return {
        'loss': total_loss / len(dataset),
        'mae': mae,
        'rmse': rmse
    }

def main():
    # Load prepared data
    logging.info("Loading prepared data...")
    data = torch.load('data/training_data.pt')
    features = data['features']
    labels = data['labels']
    price_scaler = data['price_scaler']
    
    # Parameters for rolling window validation
    seq_length = 10  # 2 weeks of trading days
    n_splits = 5  # Number of validation splits
    min_train_size = 100  # Minimum size of training window
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=20)
    
    # Store metrics for each fold
    fold_metrics = []
    
    # Rolling window validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
        logging.info(f"\nFold {fold + 1}/{n_splits}")
        
        # Get train/val split for this fold
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]
        
        logging.info(f"Training window: {len(train_features)} days")
        logging.info(f"Validation window: {len(val_features)} days")
        
        # Create datasets
        train_dataset = OptionsDataset(
            features=train_features,
            labels=train_labels,
            seq_length=seq_length
        )
        
        val_dataset = OptionsDataset(
            features=val_features,
            labels=val_labels,
            seq_length=seq_length
        )
        
        # Create model for this fold
        config = OptionsGPTConfig(
            n_embd=256,
            n_head=4,
            n_layer=2,
            n_positions=seq_length,
            n_features=features.shape[1],
            dropout=0.35
        )
        model = OptionsGPT(config)
        
        # Train on this fold
        trainer = OptionsTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=32,
            learning_rate=3e-5,
            num_epochs=100,
            warmup_steps=100,
            weight_decay=0.1,
            device="cpu",
            checkpoint_dir=f"checkpoints/fold_{fold+1}"
        )
        
        trainer.train()
        
        # Evaluate on validation set
        val_metrics = evaluate_window(model, val_dataset)
        fold_metrics.append(val_metrics)
        
        logging.info(f"Fold {fold + 1} Validation Metrics:")
        logging.info(f"Loss: {val_metrics['loss']:.4f}")
        logging.info(f"MAE: {val_metrics['mae']:.4f}")
        logging.info(f"RMSE: {val_metrics['rmse']:.4f}")
    
    # Calculate and log average metrics across folds
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in ['loss', 'mae', 'rmse']
    }
    std_metrics = {
        metric: np.std([fold[metric] for fold in fold_metrics])
        for metric in ['loss', 'mae', 'rmse']
    }
    
    logging.info("\nAverage Metrics Across All Folds:")
    for metric in ['loss', 'mae', 'rmse']:
        logging.info(f"{metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
    
    # Train final model on most recent window for future predictions
    recent_features = features[-min_train_size:]
    recent_labels = labels[-min_train_size:]
    
    recent_dataset = OptionsDataset(
        features=recent_features,
        labels=recent_labels,
        seq_length=seq_length
    )
    
    final_model = OptionsGPT(config)
    final_trainer = OptionsTrainer(
        model=final_model,
        train_dataset=recent_dataset,
        val_dataset=recent_dataset,
        batch_size=32,
        learning_rate=3e-5,
        num_epochs=100,
        warmup_steps=100,
        weight_decay=0.1,
        device="cpu",
        checkpoint_dir="checkpoints/final_model"
    )
    
    final_trainer.train()
    
    # Calculate target prediction day
    last_date = pd.Timestamp.now() + pd.Timedelta(days=1)
    while last_date.dayofweek >= 5:  # Skip weekends
        last_date += pd.Timedelta(days=1)
    logging.info(f"\nTarget prediction day: {last_date.date()}")

if __name__ == "__main__":
    main() 