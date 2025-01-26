#!/usr/bin/env python3
"""
Script to train models for all stocks in the training data.
"""

import os
import logging
from datetime import datetime
import torch
import yaml

from src.training.config import TrainingConfiguration
from src.training.train import prepare_data
from src.training.model import OptionsGPT, OptionsGPTConfig
from src.training.trainer import OptionsTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_stock_model(symbol: str, config: TrainingConfiguration):
    """Train a model for a single stock."""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*50}")
        logger.info("Preparing datasets...")
        
        # Setup paths
        data_path = os.path.join("data/training", f"{symbol}_training_data.parquet")
        model_dir = os.path.join("models", symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = prepare_data(
            data_path=data_path,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seq_length=config.data.seq_length,
            device=config.training.device
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        logger.info(f"Initializing model for {symbol}...")
        # Create model config from training config
        model_config = OptionsGPTConfig(
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout
        )
        
        # Initialize model with config
        model = OptionsGPT(model_config)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize trainer
        trainer = OptionsTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            num_epochs=config.training.num_epochs,
            device=config.training.device,
            checkpoint_dir=os.path.join(model_dir, "checkpoints"),
            stock_symbol=symbol  # Add stock symbol for logging
        )
        
        # Train model
        logger.info(f"Starting training for {symbol}...")
        trainer.train()
        
        # Evaluate on test set
        logger.info(f"\nEvaluating {symbol} on test set...")
        test_metrics = trainer.evaluate(test_dataset)
        logger.info(f"Test metrics for {symbol}:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"{'='*50}\n")
        
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
    return model

def main():
    """Main function to train models for all stocks."""
    # List of stock symbols
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "V", "WMT"
    ]
    
    # Load configuration
    config_path = "configs/training_config.yaml"
    config = TrainingConfiguration.from_yaml(config_path)
    
    # Train model for each stock
    for symbol in symbols:
        try:
            train_stock_model(symbol, config)
        except Exception as e:
            logger.error(f"Failed to train model for {symbol}: {e}")
            continue

if __name__ == "__main__":
    main() 