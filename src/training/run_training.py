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

def validate_data(train_dataset, val_dataset, test_dataset, logger):
    """
    Validate the prepared datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        logger: Logger instance
    """
    logger.info("\nValidating prepared data:")
    logger.info("-" * 50)
    
    # Check dataset sizes
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Train samples: {len(train_dataset)} ({len(train_dataset)/total_samples:.1%})")
    logger.info(f"Val samples: {len(val_dataset)} ({len(val_dataset)/total_samples:.1%})")
    logger.info(f"Test samples: {len(test_dataset)} ({len(test_dataset)/total_samples:.1%})")
    
    # Check feature dimensions
    features, labels = train_dataset[0]
    logger.info(f"\nFeature dimensions: {features.shape}")
    logger.info(f"Label dimensions: {labels.shape}")
    
    # Analyze label distribution
    train_labels = torch.stack([labels for _, labels in train_dataset])
    val_labels = torch.stack([labels for _, labels in val_dataset])
    test_labels = torch.stack([labels for _, labels in test_dataset])
    
    # Direction distribution
    train_direction = train_labels[:, 0]
    val_direction = val_labels[:, 0]
    test_direction = test_labels[:, 0]
    
    logger.info("\nDirection label distribution:")
    logger.info(f"Train - Down (-1): {(train_direction == -1).float().mean():.1%}, "
               f"Neutral (0): {(train_direction == 0).float().mean():.1%}, "
               f"Up (1): {(train_direction == 1).float().mean():.1%}")
    logger.info(f"Val   - Down (-1): {(val_direction == -1).float().mean():.1%}, "
               f"Neutral (0): {(val_direction == 0).float().mean():.1%}, "
               f"Up (1): {(val_direction == 1).float().mean():.1%}")
    logger.info(f"Test  - Down (-1): {(test_direction == -1).float().mean():.1%}, "
               f"Neutral (0): {(test_direction == 0).float().mean():.1%}, "
               f"Up (1): {(test_direction == 1).float().mean():.1%}")
    
    # Returns statistics
    train_returns = train_labels[:, 1]
    val_returns = val_labels[:, 1]
    test_returns = test_labels[:, 1]
    
    logger.info("\nReturns statistics:")
    logger.info(f"Train - Mean: {train_returns.mean():.4f}, Std: {train_returns.std():.4f}")
    logger.info(f"Val   - Mean: {val_returns.mean():.4f}, Std: {val_returns.std():.4f}")
    logger.info(f"Test  - Mean: {test_returns.mean():.4f}, Std: {test_returns.std():.4f}")
    
    # Volatility statistics
    train_vol = train_labels[:, 2]
    val_vol = val_labels[:, 2]
    test_vol = test_labels[:, 2]
    
    logger.info("\nVolatility statistics:")
    logger.info(f"Train - Mean: {train_vol.mean():.4f}, Std: {train_vol.std():.4f}")
    logger.info(f"Val   - Mean: {val_vol.mean():.4f}, Std: {val_vol.std():.4f}")
    logger.info(f"Test  - Mean: {test_vol.mean():.4f}, Std: {test_vol.std():.4f}")
    
    # Check for NaN/Inf values
    def check_invalid_values(tensor, name):
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"\nFound invalid values in {name}:")
            logger.warning(f"NaN count: {nan_count}")
            logger.warning(f"Inf count: {inf_count}")
    
    logger.info("\nChecking for invalid values...")
    for dataset, name in [(train_dataset, "train"), (val_dataset, "val"), (test_dataset, "test")]:
        features, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset))))
        check_invalid_values(features, f"{name} features")
        check_invalid_values(labels, f"{name} labels")
    
    logger.info("\nData validation completed.")
    logger.info("-" * 50)

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
    
    # Validate prepared data
    validate_data(train_dataset, val_dataset, test_dataset, logger)
    
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
        checkpoint_dir=config.training.checkpoint_dir,
        log_interval=100
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
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