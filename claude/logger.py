"""
Centralized logging configuration for Claude Code.

This module provides a consistent logging setup across the application
with proper debug level configuration and formatting.
"""

import logging
import os
from pathlib import Path


def setup_logger(name: str, level: str = "DEBUG") -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: Name of the logger (usually __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set logger level
        log_level = getattr(logging, level.upper(), logging.DEBUG)
        logger.setLevel(log_level)
        
        # Create console handler with debug level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create file handler for app.log in current working directory
        log_file = Path.cwd() / "app.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Prevent logging from propagating to the root logger
        logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default debug configuration.
    
    Args:
        name: Name of the logger (usually __name__ from calling module)
    
    Returns:
        Configured logger instance
    """
    return setup_logger(name, level="ERROR")
