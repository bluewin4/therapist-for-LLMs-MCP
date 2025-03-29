"""
Refusal detector for the MCP Therapist.

This module provides a detector that identifies when the LLM
is refusing to respond to user requests.
"""

import re
from typing import Dict, List, Optional, Tuple

from mcp_therapist.config.settings import settings
from mcp_therapist.core.detectors.base import BaseDetector, DetectionResult
from mcp_therapist.core.detectors.registry import register_detector
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType, RutAnalysisResult
from mcp_therapist.utils.logging import logger


@register_detector
class RefusalDetector(BaseDetector):
    """Detector for identifying refusal patterns in LLM responses."""
    
    def __init__(self, min_messages: int = 2, refusal_keywords: Optional[List[str]] = None):
        """Initialize the refusal detector.
        
        Args:
            min_messages: Minimum number of messages required for detection.
            refusal_keywords: Keywords that indicate refusal. If None, uses settings.
        """
        super().__init__()
        self.min_messages = min_messages
        self.refusal_keywords = refusal_keywords or settings.REFUSAL_KEYWORDS
        
        # Compile patterns for better performance
        self.refusal_patterns = [
            re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for keyword in self.refusal_keywords
        ]
    
    @property
    def detector_type(self) -> str:
        """Return the type of detector."""
        return "refusal"
    
    @property
    def rut_type(self) -> RutType:
        """Return the type of rut this detector identifies."""
        return RutType.REFUSAL
    
    def analyze(self, conversation: Conversation) -> RutAnalysisResult:
        """Analyze the conversation for signs of refusal patterns.
        
        Args:
            conversation: The conversation to analyze.
            
        Returns:
            RutAnalysisResult with refusal detection data.
        """
        # Call analyze_conversation to implement the detection logic
        detection_result = self.analyze_conversation(conversation)
        
        # Convert DetectionResult to RutAnalysisResult
        return RutAnalysisResult(
            conversation_id=conversation.id,
            rut_detected=detection_result.rut_detected,
            rut_type=detection_result.rut_type,
            confidence=detection_result.confidence,
            evidence=detection_result.evidence
        )
    
    def analyze_conversation(self, conversation: Conversation) -> DetectionResult:
        """Analyze a conversation for refusal patterns.
        
        Args:
            conversation: The conversation to analyze.
            
        Returns:
            A detection result indicating whether refusal was detected.
        """
        # Check if there are enough messages
        if len(conversation.messages) < self.min_messages:
            return DetectionResult(
                rut_detected=False,
                rut_type=RutType.REFUSAL,
                confidence=0.0,
                evidence={"details": "Not enough messages for refusal detection"}
            )
        
        # Get recent assistant messages
        assistant_messages = [
            msg for msg in conversation.messages[-5:] 
            if msg.role == MessageRole.ASSISTANT
        ]
        
        if not assistant_messages:
            return DetectionResult(
                rut_detected=False,
                rut_type=RutType.REFUSAL,
                confidence=0.0,
                evidence={"details": "No assistant messages found"}
            )
        
        # Check the last assistant message for refusal patterns
        last_msg = assistant_messages[-1]
        
        # Count the number of refusal keywords in the message
        refusals = []
        content = last_msg.content.lower()
        
        for pattern in self.refusal_patterns:
            matches = pattern.findall(content)
            if matches:
                refusals.extend(matches)
        
        # If refusals are found, report as a rut
        if refusals:
            evidence = {
                "details": f"Refusal patterns detected: {', '.join(refusals)}",
                "message_idx": conversation.messages.index(last_msg),
                "refusal_patterns": refusals
            }
            
            # Calculate confidence based on number of refusal patterns
            # More refusal patterns = higher confidence
            confidence = min(0.6 + 0.1 * len(refusals), 0.9)
            
            return DetectionResult(
                rut_detected=True,
                rut_type=RutType.REFUSAL,
                confidence=confidence,
                evidence=evidence
            )
            
        # No refusal patterns found
        return DetectionResult(
            rut_detected=False,
            rut_type=RutType.REFUSAL,
            confidence=0.0,
            evidence={"details": "No refusal patterns detected"}
        ) 