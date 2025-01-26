"""
Script to run model training with configuration and monitoring.
"""

import os
import logging
import argparse
from datetime import datetime

import torch
import wandb
from transformers import GPT2Config
from torch.utils.tensorboard import SummaryWriter

from .model import OptionsGPT
from .trainer import OptionsTrainer
from .train import prepare_data
from .config import TrainingConfiguration

def setup_logging(log_dir: str) -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_model(config: TrainingConfiguration, data_path: str):
    """
    Train the model with given configuration.
    
    Args:
        config: Training configuration
        data_path: Path to data file
    """
    # Set up logging
    setup_logging(config.training.log_dir)
    logger = logging.getLogger(__name__)
    
    # Create log directory if it doesn't exist
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=config.training.log_dir)
    
    # Initialize wandb if enabled
    if config.monitoring.use_wandb:
        wandb.init(
            project=config.monitoring.project_name,
            config={
                'data': config.data.__dict__,
                'model': config.model.__dict__,
                'training': config.training.__dict__,
                'early_stopping': config.early_stopping.__dict__
            },
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing data...")
    train_dataset, val_dataset, test_dataset = prepare_data(
        data_path,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seq_length=config.data.seq_length
    )
    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    
    # Create model
    logger.info("Creating model...")
    model_config = GPT2Config(
        n_embd=config.model.hidden_size,
        n_layer=config.model.num_layers,
        n_head=config.model.num_heads,
        resid_pdrop=config.model.dropout,
        summary_first_dropout=config.model.dropout
    )
    model = OptionsGPT(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = OptionsTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        device=device,
        checkpoint_dir=config.training.checkpoint_dir
    )
    
    # Training loop with monitoring
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch()
        logger.info("Training metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            writer.add_scalar(f"train/{metric}", value, epoch)
            if config.monitoring.use_wandb:
                wandb.log({f"train/{metric}": value}, step=epoch)
        
        # Validate
        val_metrics = trainer.evaluate(trainer.val_loader)
        logger.info("Validation metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            writer.add_scalar(f"val/{metric}", value, epoch)
            if config.monitoring.use_wandb:
                wandb.log({f"val/{metric}": value}, step=epoch)
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss - config.early_stopping.min_delta:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            logger.info(f"Validation loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{config.early_stopping.patience}")
            
        if patience_counter >= config.early_stopping.patience:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs. "
                f"Best validation loss: {best_val_loss:.4f}"
            )
            break
            
        # Save checkpoint only if validation loss improved
        if patience_counter == 0:
            trainer.save_checkpoint(val_metrics, epoch + 1)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(trainer.test_loader)
    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
        if config.monitoring.use_wandb:
            wandb.log({f"test/{metric}": value})
    
    # Close tensorboard writer
    writer.close()
    
    # Close wandb
    if config.monitoring.use_wandb:
        wandb.finish()

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description="Train options trading model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainingConfiguration.from_yaml(args.config)
    
    # Train model
    train_model(config, args.data_path)

if __name__ == "__main__":
    main() 