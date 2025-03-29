"""
Registry for rut detectors.

This module provides a registration system for rut detectors, allowing them to be
dynamically registered and retrieved by name or rut type.
"""

from typing import Dict, List, Optional, Type

from mcp_therapist.models.conversation import RutType
from mcp_therapist.utils.logging import logger

# Avoid circular imports with forward references
from .base import RutDetector

# Registry of detectors by name and by rut type
_detector_registry: Dict[str, Type[RutDetector]] = {}
_detector_by_rut_type: Dict[RutType, Type[RutDetector]] = {}


def register_detector(detector_class: Type[RutDetector]) -> Type[RutDetector]:
    """Register a detector class in the registry.
    
    This function can be used as a decorator on detector classes to register them
    automatically.
    
    Args:
        detector_class: The detector class to register.
        
    Returns:
        The same detector class (to allow use as a decorator).
        
    Raises:
        ValueError: If a detector with the same name is already registered.
    """
    detector_name = detector_class.__name__
    
    # Check if already registered
    if detector_name in _detector_registry:
        logger.warning(f"Detector {detector_name} already registered. Overwriting.")
    
    # Create a temporary instance to get the rut type
    temp_instance = detector_class()
    rut_type = temp_instance.rut_type
    
    # Register by name and rut type
    _detector_registry[detector_name] = detector_class
    _detector_by_rut_type[rut_type] = detector_class
    
    logger.debug(f"Registered detector {detector_name} for rut type {rut_type.value}")
    
    return detector_class


def get_detector(name_or_type: str | RutType) -> Optional[Type[RutDetector]]:
    """Get a detector class by name or rut type.
    
    Args:
        name_or_type: Either a detector class name or a RutType.
        
    Returns:
        The detector class if found, None otherwise.
    """
    if isinstance(name_or_type, str):
        return _detector_registry.get(name_or_type)
    elif isinstance(name_or_type, RutType):
        return _detector_by_rut_type.get(name_or_type)
    else:
        logger.error(f"Invalid detector identifier: {name_or_type}")
        return None


def get_all_detectors() -> List[Type[RutDetector]]:
    """Get all registered detector classes.
    
    Returns:
        A list of all registered detector classes.
    """
    return list(_detector_registry.values()) 