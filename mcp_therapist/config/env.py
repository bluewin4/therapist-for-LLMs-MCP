"""
Environment variable loader for MCP Therapist.

This module handles loading environment variables from .env files,
making them available to the application.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_dotenv(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the environment file, relative to project root
        
    Returns:
        Dictionary of loaded environment variables
    """
    loaded_vars = {}
    env_path = BASE_DIR / env_file
    
    if not env_path.exists():
        return loaded_vars
    
    # Read the .env file
    with open(env_path, "r") as f:
        content = f.read()
    
    # Parse each line
    for line in content.splitlines():
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        
        # Extract key and value
        match = re.match(r'^([A-Za-z0-9_]+)=(.*)$', line)
        if match:
            key, value = match.groups()
            # Remove quotes if present
            value = value.strip('"\'')
            # Set environment variable if not already set
            if key not in os.environ:
                os.environ[key] = value
                loaded_vars[key] = value
    
    return loaded_vars


def get_env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get string environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Boolean interpretation of environment variable
    """
    value = os.environ.get(key, "").lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "y", "on")


def get_env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Integer interpretation of environment variable
    """
    value = os.environ.get(key)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    """
    Get float environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Float interpretation of environment variable
    """
    value = os.environ.get(key)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Load environment variables from files
def load_environment():
    """Load environment variables from all .env files."""
    # Order of precedence: .env.local (highest) -> .env (lowest)
    env_files = [
        ".env",
        ".env.local",
    ]
    
    for env_file in env_files:
        load_dotenv(env_file)


# Load environment on module import
load_environment() 