"""
Detector modules for identifying conversation 'ruts'.

This package contains detector implementations that analyze conversation
history to identify patterns indicating the LLM is stuck in an unhelpful state.
"""

from .base import RutDetector, DetectionResult
from .registry import register_detector, get_detector, get_all_detectors
from .manager import DetectionManager

# Import all detectors for registration
from .init_detectors import register_all_detectors

__all__ = [
    'RutDetector',
    'DetectionResult',
    'register_detector',
    'get_detector',
    'get_all_detectors',
    'DetectionManager',
] 