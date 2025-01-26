"""
Configuration loader utility.
"""

import os
from pathlib import Path
from typing import Dict, Any

import yaml

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Get default config path
        root_dir = Path(__file__).resolve().parent.parent.parent
        config_path = os.path.join(root_dir, 'configs', 'default_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_feature_groups(extractor_type: str) -> Dict[str, list]:
    """
    Get feature groups configuration for a specific extractor type.
    
    Args:
        extractor_type: Type of feature extractor ('options' or 'technical')
        
    Returns:
        Dictionary mapping feature group names to lists of feature names
    """
    config = load_config()
    return config['feature_extractors'][extractor_type]['feature_groups'] 