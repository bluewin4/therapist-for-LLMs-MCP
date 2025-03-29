"""
Base detector classes and interfaces for rut detection.

This module defines the base classes and interfaces for all rut detectors,
providing a common API for detector implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType, RutAnalysisResult
from mcp_therapist.utils.logging import logger
from mcp_therapist.config.settings import settings


class DetectionResult:
    """Result of a rut detection analysis.
    
    This class represents the result of a rut detector's analysis, including whether
    a rut was detected, the type of rut, confidence scores, and supporting evidence.
    """
    
    def __init__(
        self,
        rut_detected: bool,
        rut_type: RutType,
        confidence: float,
        evidence: Union[Dict, List[str]],
    ):
        """Initialize a detection result.
        
        Args:
            rut_detected: Whether a rut was detected.
            rut_type: The type of rut detected.
            confidence: Confidence score (0.0 to 1.0) for the detection.
            evidence: Evidence supporting the detection result.
                      Can be a dictionary of analysis details or a list of string explanations.
        """
        self.rut_detected = rut_detected
        self.rut_type = rut_type
        self.confidence = confidence
        self.evidence = evidence if isinstance(evidence, dict) else {"details": evidence}
        self.timestamp = None  # Will be set when added to conversation
    
    def to_dict(self) -> Dict:
        """Convert the detection result to a dictionary.
        
        Returns:
            Dictionary representation of the detection result.
        """
        return {
            "rut_detected": self.rut_detected,
            "rut_type": self.rut_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DetectionResult":
        """Create a detection result from a dictionary.
        
        Args:
            data: Dictionary representation of a detection result.
            
        Returns:
            DetectionResult instance.
        """
        result = cls(
            rut_detected=data["rut_detected"],
            rut_type=RutType(data["rut_type"]),
            confidence=data["confidence"],
            evidence=data["evidence"]
        )
        result.timestamp = data.get("timestamp")
        return result


class BaseDetector(ABC):
    """Base class for all rut detectors.
    
    This abstract base class defines the interface that all rut detectors must implement.
    It provides common utility methods and enforces a standard API for detector implementations.
    """
    
    def __init__(self):
        """Initialize the base detector."""
        self.logger = logger
    
    @property
    @abstractmethod
    def detector_type(self) -> str:
        """Get the type of this detector."""
        pass
    
    @property
    @abstractmethod
    def rut_type(self) -> RutType:
        """Get the rut type that this detector identifies."""
        pass
    
    @abstractmethod
    def analyze(self, conversation: Conversation, window_size: Optional[int] = None) -> DetectionResult:
        """Analyze a conversation for signs of a conversational rut.
        
        This is the main method that subclasses must implement to perform their
        specific rut detection logic.
        
        Args:
            conversation: The conversation to analyze.
            window_size: Optional window size for the analysis.
                
        Returns:
            DetectionResult with the analysis results.
        """
        pass
    
    def get_assistant_messages(self, messages: List[Message]) -> List[Message]:
        """Filter a list of messages to include only assistant messages.
        
        Args:
            messages: List of messages to filter.
            
        Returns:
            List of assistant messages.
        """
        return [msg for msg in messages if msg.role == MessageRole.ASSISTANT]
    
    def get_user_messages(self, messages: List[Message]) -> List[Message]:
        """Filter a list of messages to include only user messages.
        
        Args:
            messages: List of messages to filter.
            
        Returns:
            List of user messages.
        """
        return [msg for msg in messages if msg.role == MessageRole.USER]
    
    def get_message_pairs(
        self, messages: List[Message]
    ) -> List[tuple[Message, Message]]:
        """Extract user-assistant message pairs from a list of messages.
        
        Args:
            messages: List of messages to process.
            
        Returns:
            List of (user_message, assistant_message) pairs.
        """
        pairs = []
        
        for i in range(len(messages) - 1):
            if (messages[i].role == MessageRole.USER and
                messages[i + 1].role == MessageRole.ASSISTANT):
                pairs.append((messages[i], messages[i + 1]))
        
        return pairs
        
    def log_detection(self, result: DetectionResult) -> None:
        """Log a detection result.
        
        Args:
            result: The detection result to log.
        """
        detector_name = self.detector_type
        if result.rut_detected:
            self.logger.info(
                f"Detector {detector_name} found {result.rut_type.value} "
                f"with confidence {result.confidence:.2f}"
            )
        else:
            self.logger.debug(
                f"Detector {detector_name} did not detect a rut "
                f"(confidence: {result.confidence:.2f})"
            )


class RutDetector:
    """Base class for all rut detectors."""
    
    @abstractmethod
    def analyze(self, conversation: Conversation) -> RutAnalysisResult:
        """Analyze a conversation for specific rut patterns.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            A RutAnalysisResult containing detection information
        """
        pass

    def get_settings(self):
        """Get the settings for the detector."""
        return settings

    def set_settings(self, new_settings: Dict[str, Any]):
        """Set the settings for the detector."""
        settings.update(new_settings) 