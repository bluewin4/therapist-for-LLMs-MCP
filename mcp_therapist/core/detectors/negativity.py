"""
Negativity detector for MCP Therapist.

This module provides a detector that identifies patterns of
negative sentiment in conversations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime

from mcp_therapist.config.settings import settings
from mcp_therapist.core.detectors.base import BaseDetector, DetectionResult
from mcp_therapist.core.detectors.registry import register_detector
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType, RutAnalysisResult
from mcp_therapist.utils.sentiment import sentiment_analyzer
from mcp_therapist.utils.logging import logger


@register_detector
class NegativityDetector(BaseDetector):
    """Detector for identifying patterns of negative sentiment in conversations."""

    def __init__(self, min_messages: int = 3, negativity_threshold: float = 0.7):
        """Initialize the negativity detector.
        
        Args:
            min_messages: Minimum number of messages required for detection.
            negativity_threshold: Threshold for determining negative sentiment.
        """
        super().__init__()
        self.min_messages = min_messages
        self.negativity_threshold = getattr(settings, "NEGATIVITY_THRESHOLD", negativity_threshold)
        logger.info(f"Negativity detector initialized with threshold {self.negativity_threshold}")
    
    @property
    def detector_type(self) -> str:
        """Return the type of detector."""
        return "negativity"
    
    @property
    def rut_type(self) -> RutType:
        """Return the type of rut this detector identifies."""
        return RutType.NEGATIVITY
    
    def analyze(self, conversation: Conversation) -> RutAnalysisResult:
        """Analyze the conversation for signs of negative sentiment patterns.
        
        Args:
            conversation: The conversation to analyze.
            
        Returns:
            RutAnalysisResult with negativity detection data.
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
        """Analyze a conversation for patterns of negative sentiment.
        
        The detector looks for:
        1. Consecutive messages with negative sentiment
        2. An overall negative trend in the conversation
        3. Mismatch between user and assistant sentiment
        
        Args:
            conversation: The conversation to analyze.
            
        Returns:
            A detection result indicating whether negative sentiment was detected.
        """
        # Check if there are enough messages
        if len(conversation.messages) < self.min_messages:
            return DetectionResult(
                rut_detected=False,
                rut_type=RutType.NEGATIVITY,
                confidence=0.0,
                evidence={"details": "Not enough messages for negativity detection"}
            )
            
        # Get recent messages
        recent_messages = conversation.messages[-min(10, len(conversation.messages)):]
        
        # Analyze sentiment for each message
        sentiment_scores = []
        for msg in recent_messages:
            score = sentiment_analyzer.analyze_text(msg.content)
            sentiment_scores.append({
                "role": msg.role,
                "score": score,
                "content": msg.content[:100]  # Include truncated content for reference
            })
        
        # Check for consecutive negative messages
        consecutive_negative = self._check_consecutive_negative(sentiment_scores)
        
        # Check for sentiment mismatch between user and assistant
        sentiment_mismatch = self._check_sentiment_mismatch(sentiment_scores)
        
        # Check for negative trend
        negative_trend = self._check_sentiment_trend(sentiment_scores)
        
        # Combine evidence
        evidence = {
            "sentiment_scores": sentiment_scores,
            "consecutive_negative": consecutive_negative,
            "sentiment_mismatch": sentiment_mismatch,
            "negative_trend": negative_trend
        }
        
        # Calculate overall confidence
        confidence = 0.0
        reasons = []
        
        if consecutive_negative["detected"]:
            confidence = max(confidence, consecutive_negative["confidence"])
            reasons.append(f"Consecutive negative messages: {consecutive_negative['count']}")
            
        if sentiment_mismatch["detected"]:
            confidence = max(confidence, sentiment_mismatch["confidence"])
            reasons.append(f"Sentiment mismatch between user and assistant")
            
        if negative_trend["detected"]:
            confidence = max(confidence, negative_trend["confidence"])
            reasons.append(f"Negative trend in conversation")
        
        if confidence > 0:
            evidence["details"] = "; ".join(reasons)
        else:
            evidence["details"] = "No negative sentiment patterns detected"
        
        # Determine if negativity is detected
        rut_detected = confidence >= self.negativity_threshold
        
        return DetectionResult(
            rut_detected=rut_detected,
            rut_type=RutType.NEGATIVITY,
            confidence=confidence,
            evidence=evidence
        )
    
    def _check_consecutive_negative(self, sentiment_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for consecutive messages with negative sentiment.
        
        Args:
            sentiment_scores: List of sentiment scores for messages.
            
        Returns:
            Detection results for consecutive negative messages.
        """
        max_consecutive = 0
        current_consecutive = 0
        
        for score_data in sentiment_scores:
            if score_data["score"] < 0.4:  # Negative sentiment threshold
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # Calculate confidence based on number of consecutive negative messages
        confidence = 0.0
        if max_consecutive >= 3:
            confidence = 0.8
        elif max_consecutive == 2:
            confidence = 0.6
        elif max_consecutive == 1:
            confidence = 0.3
        
        return {
            "detected": max_consecutive >= 2,
            "count": max_consecutive,
            "confidence": confidence
        }
    
    def _check_sentiment_mismatch(self, sentiment_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for mismatch between user and assistant sentiment.
        
        Args:
            sentiment_scores: List of sentiment scores for messages.
            
        Returns:
            Detection results for sentiment mismatch.
        """
        # Separate user and assistant scores
        user_scores = [data["score"] for data in sentiment_scores if data["role"] == MessageRole.USER]
        assistant_scores = [data["score"] for data in sentiment_scores if data["role"] == MessageRole.ASSISTANT]
        
        if not user_scores or not assistant_scores:
            return {"detected": False, "confidence": 0.0}
        
        # Calculate average sentiment for user and assistant
        avg_user = sum(user_scores) / len(user_scores)
        avg_assistant = sum(assistant_scores) / len(assistant_scores)
        
        # Check if user is positive but assistant is negative
        user_positive = avg_user > 0.6
        assistant_negative = avg_assistant < 0.4
        
        # Or if user is neutral but assistant is very negative
        user_neutral = 0.4 <= avg_user <= 0.6
        assistant_very_negative = avg_assistant < 0.3
        
        mismatch_detected = (user_positive and assistant_negative) or (user_neutral and assistant_very_negative)
        
        # Calculate confidence based on the difference in sentiment
        confidence = min(abs(avg_user - avg_assistant), 1.0) if mismatch_detected else 0.0
        
        return {
            "detected": mismatch_detected,
            "user_avg": avg_user,
            "assistant_avg": avg_assistant,
            "confidence": confidence
        }
    
    def _check_sentiment_trend(self, sentiment_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for a negative trend in sentiment over the conversation.
        
        Args:
            sentiment_scores: List of sentiment scores for messages.
            
        Returns:
            Detection results for negative sentiment trend.
        """
        if len(sentiment_scores) < 4:
            return {"detected": False, "confidence": 0.0}
        
        # Extract scores in sequence
        scores = [data["score"] for data in sentiment_scores]
        
        # Calculate sentiment at beginning and end of conversation
        start_sentiment = sum(scores[:len(scores)//2]) / (len(scores)//2)
        end_sentiment = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        
        # Check for significant negative shift
        shift = end_sentiment - start_sentiment
        significant_negative_shift = shift < -0.2
        
        # Calculate confidence based on the magnitude of the shift
        confidence = min(abs(shift) * 2, 0.9) if significant_negative_shift else 0.0
        
        return {
            "detected": significant_negative_shift,
            "start_sentiment": start_sentiment,
            "end_sentiment": end_sentiment,
            "shift": shift,
            "confidence": confidence
        } 