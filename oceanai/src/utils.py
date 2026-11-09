"""
Oceanographic AI Platform - Utilities & Helper Functions
========================================================
Handles logging, configuration loading, seed fixing, spatial computations,
and environment setup.
"""

import logging
import os
from pathlib import Path
import random
import numpy as np
import yaml

# Headless Matplotlib configuration
LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".matplotlib_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(LOCAL_CACHE_DIR)


def setup_logger(name: str = "oceanai", log_file: Path | None = None) -> logging.Logger:
    """Configure a structured logger with both console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger


def set_seed(seed: int = 42):
    """Set random seed across standard library, numpy, and environment for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: Path | None = None) -> dict:
    """Load the project YAML configuration file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "regions.yaml"
        
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_oceanai_dir() -> Path:
    """Return the oceanai subfolder directory."""
    return Path(__file__).resolve().parent.parent
