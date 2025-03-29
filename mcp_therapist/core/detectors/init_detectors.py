"""
Detector initialization module.

This module initializes and registers all available detectors.
Import this module to ensure all detectors are registered correctly.
"""

from mcp_therapist.core.detectors.registry import register_detector
from mcp_therapist.utils.logging import logger

# Import all detector classes
from mcp_therapist.core.detectors.repetition import RepetitionDetector
from mcp_therapist.core.detectors.stagnation import StagnationDetector
from mcp_therapist.core.detectors.refusal import RefusalDetector
from mcp_therapist.core.detectors.negativity import NegativityDetector
from mcp_therapist.core.detectors.topic_fixation import TopicFixationDetector
from mcp_therapist.core.detectors.contradiction import ContradictionDetector


# Register all detectors
def register_all_detectors():
    """Register all available detectors with the registry."""
    logger.info("Registering detectors...")
    
    # Register core detectors
    register_detector(RepetitionDetector)
    register_detector(StagnationDetector)
    register_detector(RefusalDetector)
    register_detector(NegativityDetector)
    register_detector(TopicFixationDetector)
    register_detector(ContradictionDetector)
    
    logger.info("All detectors registered successfully")


# Auto-register detectors when module is imported
register_all_detectors() 