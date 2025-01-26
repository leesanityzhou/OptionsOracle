"""
Configuration classes for model training.
"""

import os
from dataclasses import dataclass
from typing import Optional

import yaml

@dataclass
class DataConfig:
    """Data configuration."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seq_length: int = 30

    def validate(self):
        """Validate data configuration."""
        if not (0 <= self.train_ratio <= 1 and 0 <= self.val_ratio <= 1 and 0 <= self.test_ratio <= 1):
            raise ValueError("All ratios must be between 0 and 1")
            
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1")
            
        if self.seq_length <= 0:
            raise ValueError("Sequence length must be positive")

@dataclass
class ModelConfig:
    """Model configuration."""
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    def validate(self):
        """Validate model configuration."""
        if self.hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
            
        if self.num_layers <= 0:
            raise ValueError("Number of layers must be positive")
            
        if self.num_heads <= 0:
            raise ValueError("Number of heads must be positive")
            
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be between 0 and 1")

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def validate(self):
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
            
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
            
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")

@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    patience: int = 5
    min_delta: float = 1e-4

    def validate(self):
        """Validate early stopping configuration."""
        if self.patience <= 0:
            raise ValueError("Patience must be positive")
            
        if self.min_delta < 0:
            raise ValueError("Min delta must be non-negative")

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    use_wandb: bool = False
    project_name: str = "options-oracle"

    def validate(self):
        """Validate monitoring configuration."""
        if not isinstance(self.use_wandb, bool):
            raise ValueError("use_wandb must be a boolean")
            
        if not self.project_name:
            raise ValueError("Project name cannot be empty")

@dataclass
class TrainingConfiguration:
    """Complete training configuration."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    monitoring: MonitoringConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfiguration":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            TrainingConfiguration object
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        early_stopping_config = EarlyStoppingConfig(**config_dict['early_stopping'])
        monitoring_config = MonitoringConfig(**config_dict['monitoring'])
        
        config = cls(
            data=data_config,
            model=model_config,
            training=training_config,
            early_stopping=early_stopping_config,
            monitoring=monitoring_config
        )
        
        config.validate()
        return config
    
    def validate(self):
        """Validate all configurations."""
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.early_stopping.validate()
        self.monitoring.validate()
        
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'early_stopping': self.early_stopping.__dict__,
            'monitoring': self.monitoring.__dict__
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False) 