"""
Script for training the options trading model.
"""

import os
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Config

from .model import OptionsGPT
from .trainer import OptionsTrainer, OptionsDataset

def prepare_data(
    data_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seq_length: int = 30
) -> tuple:
    """
    Prepare data for training.
    
    Args:
        data_path: Path to data file
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seq_length: Length of sequences to create
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Raises:
        ValueError: If ratios don't sum to 1, are negative, or if sequence length is invalid
    """
    # Validate ratios
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("All ratios must be between 0 and 1")
        
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1")
    
    # Load data
    try:
        data = pd.read_parquet(data_path)
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {str(e)}")
    
    # Validate sequence length
    if seq_length <= 0:
        raise ValueError("Sequence length must be positive")
    if seq_length > len(data):
        raise ValueError("Sequence length cannot be longer than dataset")
        
    # Check required columns
    required_columns = ['date', 'direction', 'returns', 'volatility']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sort by date
    data = data.sort_values('date')
    
    # Split features and labels
    feature_cols = [col for col in data.columns if col not in ['date', 'direction', 'returns', 'volatility']]
    if not feature_cols:
        raise ValueError("No feature columns found in data")
        
    features = data[feature_cols].values
    direction_labels = data['direction'].values
    return_labels = data['returns'].values
    volatility_labels = data['volatility'].values
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Convert to tensors
    features = torch.FloatTensor(features)
    direction_labels = torch.FloatTensor(direction_labels)
    return_labels = torch.FloatTensor(return_labels)
    volatility_labels = torch.FloatTensor(volatility_labels)
    
    # Calculate split indices
    total_samples = len(features)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    # Split data
    train_features = features[:train_size]
    train_direction = direction_labels[:train_size]
    train_returns = return_labels[:train_size]
    train_volatility = volatility_labels[:train_size]
    
    val_features = features[train_size:train_size + val_size]
    val_direction = direction_labels[train_size:train_size + val_size]
    val_returns = return_labels[train_size:train_size + val_size]
    val_volatility = volatility_labels[train_size:train_size + val_size]
    
    test_features = features[train_size + val_size:]
    test_direction = direction_labels[train_size + val_size:]
    test_returns = return_labels[train_size + val_size:]
    test_volatility = volatility_labels[train_size + val_size:]
    
    # Create datasets
    train_dataset = OptionsDataset(
        train_features,
        train_direction,
        train_returns,
        train_volatility,
        seq_length
    )
    
    val_dataset = OptionsDataset(
        val_features,
        val_direction,
        val_returns,
        val_volatility,
        seq_length
    )
    
    test_dataset = OptionsDataset(
        test_features,
        test_direction,
        test_returns,
        test_volatility,
        seq_length
    )
    
    return train_dataset, val_dataset, test_dataset

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train options trading model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of data to use for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of data to use for testing")
    parser.add_argument("--seq_length", type=int, default=30, help="Length of sequences to create")
    
    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Prepare data
    logger.info("Preparing data...")
    train_dataset, val_dataset, test_dataset = prepare_data(
        args.data_path,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seq_length
    )
    logger.info(
        f"Data split: "
        f"Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )
    
    # Create model
    logger.info("Creating model...")
    config = GPT2Config(
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        resid_pdrop=args.dropout
    )
    model = OptionsGPT(config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = OptionsTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main() 