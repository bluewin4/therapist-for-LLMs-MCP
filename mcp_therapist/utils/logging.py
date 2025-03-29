"""
Logging utility for the MCP Therapist application.

This module provides a standardized logging setup for the application,
including console and file logging with configurable log levels.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional

from mcp_therapist.config.settings import settings


# Set up the logger
def setup_logger(
    name: str = "mcp_therapist",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    file_log_level: Optional[int] = None,
    console_log_level: Optional[int] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Set up a logger with console and/or file handlers.
    
    Args:
        name: The name of the logger.
        log_level: The default log level for all handlers.
        log_file: Optional path to the log file.
        console: Whether to log to console.
        file_log_level: Log level for the file handler (defaults to log_level).
        console_log_level: Log level for the console handler (defaults to log_level).
        log_format: The format string for the log messages.
        
    Returns:
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Don't propagate to parent loggers
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Remove existing handlers to avoid duplicates when called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Set up rotating file handler (max 10MB, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(file_log_level or log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_log_level or log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Get log level from string
def get_log_level(level_str: str) -> int:
    """Convert a string log level to a logging level constant.
    
    Args:
        level_str: String representation of log level (DEBUG, INFO, etc.)
        
    Returns:
        Logging level constant.
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)


# Create the default logger
def _create_default_logger() -> logging.Logger:
    """Create the default application logger.
    
    Returns:
        Configured logger instance.
    """
    # Get configuration from settings using direct attribute access
    log_level_str = settings.LOG_LEVEL
    log_file = getattr(settings, "LOG_FILE", None)  # Use getattr for optional attributes
    console = getattr(settings, "LOG_TO_CONSOLE", True)
    
    file_log_level_str = getattr(settings, "FILE_LOG_LEVEL", log_level_str)
    console_log_level_str = getattr(settings, "CONSOLE_LOG_LEVEL", log_level_str)
    
    log_level = get_log_level(log_level_str)
    file_log_level = get_log_level(file_log_level_str)
    console_log_level = get_log_level(console_log_level_str)
    
    return setup_logger(
        log_level=log_level,
        log_file=log_file,
        console=console,
        file_log_level=file_log_level,
        console_log_level=console_log_level
    )


# Export the default logger
logger = _create_default_logger()


# Convenience function to get a component-specific logger
def get_component_logger(component_name: str) -> logging.Logger:
    """Get a logger for a specific component.
    
    The component logger inherits settings from the main logger
    but allows for component-specific logging.
    
    Args:
        component_name: Name of the component.
        
    Returns:
        Component-specific logger.
    """
    return logging.getLogger(f"mcp_therapist.{component_name}") 