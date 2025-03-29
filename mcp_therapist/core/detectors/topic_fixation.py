"""
Module for detecting topic fixation in conversations.
Identifies when a conversation gets stuck on a single topic for too long.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging

from mcp_therapist.core.detectors.base import BaseDetector, DetectionResult
from mcp_therapist.models.conversation import RutType, Message, Conversation
from mcp_therapist.utils.topic import TopicAnalyzer

logger = logging.getLogger(__name__)

class TopicFixationDetector(BaseDetector):
    """
    Detector for identifying when a conversation gets stuck on a specific topic.
    Uses TF-IDF analysis to track topic similarity across messages.
    """
    
    def __init__(self, 
                min_messages: int = 5,
                window_size: int = 4,
                similarity_threshold: float = 0.6,
                confidence_threshold: float = 0.4):
        """
        Initialize the topic fixation detector.
        
        Args:
            min_messages: Minimum number of messages required before analysis
            window_size: Number of consecutive messages to analyze
            similarity_threshold: Threshold for topic similarity indicating fixation
            confidence_threshold: Minimum confidence level to report a detection
        """
        super().__init__()
        self.min_messages = min_messages
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.topic_analyzer = TopicAnalyzer()
        
    @property
    def detector_type(self) -> str:
        """Get the type of this detector."""
        return "topic_fixation"
        
    @property
    def rut_type(self) -> RutType:
        """Get the rut type that this detector identifies."""
        return RutType.TOPIC_FIXATION
        
    def _extract_message_texts(self, conversation: Conversation) -> List[str]:
        """
        Extract the text content from all messages in the conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            List of message texts
        """
        return [msg.content for msg in conversation.messages if msg.content.strip()]
        
    def detect(self, conversation: Conversation) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect topic fixation in a conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Tuple containing:
            - Boolean indicating whether topic fixation was detected
            - Confidence score (0-1)
            - Dictionary with additional detection details
        """
        # Check if there are enough messages
        if len(conversation.messages) < self.min_messages:
            logger.debug(f"Not enough messages ({len(conversation.messages)}) for topic fixation detection")
            return False, 0.0, {"reason": "Not enough messages"}
            
        # Extract message texts
        message_texts = self._extract_message_texts(conversation)
        
        # Detect topic fixation
        is_fixated, confidence, repeated_terms = self.topic_analyzer.detect_topic_fixation(
            texts=message_texts,
            window_size=self.window_size,
            similarity_threshold=self.similarity_threshold
        )
        
        # Additional details for the detection result
        details = {
            "repeated_terms": repeated_terms or [],
            "window_size": self.window_size,
            "message_count": len(message_texts)
        }
        
        # Only report fixation if confidence exceeds threshold
        if is_fixated and confidence >= self.confidence_threshold:
            logger.info(f"Topic fixation detected with confidence {confidence:.2f}. "
                       f"Repeated terms: {', '.join(repeated_terms[:5])}...")
            return True, confidence, details
        
        return False, confidence, details
    
    def analyze(self, conversation: Conversation, window_size: Optional[int] = None) -> DetectionResult:
        """
        Analyze a conversation for signs of topic fixation.
        
        Args:
            conversation: The conversation to analyze.
            window_size: Optional window size for the analysis. If provided, overrides the
                         detector's default window_size.
                
        Returns:
            DetectionResult with the analysis results.
        """
        # Use provided window_size if available, otherwise use default
        effective_window = window_size if window_size is not None else self.window_size
        
        # Run detection with appropriate window size
        detected, confidence, details = self.detect(conversation)
        
        # Create evidence dictionary
        evidence = {
            "repeated_terms": details.get("repeated_terms", []),
            "window_size": effective_window,
            "message_count": details.get("message_count", 0),
            "similarity_threshold": self.similarity_threshold,
            "confidence_threshold": self.confidence_threshold
        }
        
        # Add any additional reason if detection failed
        if "reason" in details:
            evidence["reason"] = details["reason"]
        
        return DetectionResult(
            rut_detected=detected,
            rut_type=self.rut_type,
            confidence=confidence,
            evidence=evidence
        ) 