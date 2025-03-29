"""
Detection manager for coordinating multiple detectors.

This module provides a manager that coordinates the execution of multiple
rut detectors and aggregates their results.
"""

from typing import Dict, List, Optional, Type, Union, Any

from mcp_therapist.config.settings import settings
from mcp_therapist.models.conversation import Conversation, RutAnalysisResult, RutType
from mcp_therapist.utils.logging import logger
from mcp_therapist.utils.confidence import ConfidenceScorer

from .base import DetectionResult, RutDetector
from .registry import get_all_detectors, get_detector


class DetectionManager:
    """Manager for coordinating the execution of multiple rut detectors."""
    
    def __init__(self, detectors: Optional[List[RutDetector]] = None):
        """Initialize the detection manager.
        
        Args:
            detectors: Optional list of detector instances to use. If not provided,
                all registered detectors will be instantiated.
        """
        self.logger = logger
        
        # Initialize detectors
        if detectors is not None:
            self.detectors = detectors
        else:
            self.detectors = self._initialize_detectors()
        
        # Initialize confidence scorer
        self.confidence_scorer = ConfidenceScorer()
        
        # Configure detector weights for confidence scoring
        self.detector_weights = {
            "repetition": 1.0,
            "stagnation": 0.9,
            "refusal": 0.8,
            "topic_fixation": 0.8,
            "contradiction": 0.9,
            "negativity": 0.7,
            "hallucination": 1.0,
            "other": 0.5
        }
        
        self.logger.info(f"Detection manager initialized with {len(self.detectors)} detectors")
    
    def _initialize_detectors(self) -> List[RutDetector]:
        """Initialize all registered detectors.
        
        Returns:
            List of detector instances.
        """
        detector_classes = get_all_detectors()
        detectors = []
        
        for cls in detector_classes:
            try:
                detector = cls()
                detectors.append(detector)
                self.logger.debug(f"Initialized detector: {detector}")
            except Exception as e:
                self.logger.error(f"Failed to initialize detector {cls.__name__}: {str(e)}")
        
        return detectors
    
    def analyze_conversation(self, conversation: Conversation, window_size: Optional[int] = None) -> Dict[RutType, DetectionResult]:
        """Analyze a conversation using all available detectors.
        
        Args:
            conversation: The conversation to analyze.
            window_size: Optional size of the window to analyze. If not provided,
                the default window size from settings will be used.
                
        Returns:
            Dictionary mapping rut types to detection results.
        """
        if window_size is None:
            window_size = settings.DEFAULT_WINDOW_SIZE
        
        self.logger.debug(f"Analyzing conversation {conversation.id} with window size {window_size}")
        
        results = {}
        
        for detector in self.detectors:
            try:
                # Run the detector
                result = detector.analyze(conversation, window_size)
                
                # Apply advanced confidence scoring
                self._apply_advanced_confidence(result, conversation)
                
                # Store the result
                results[result.rut_type] = result
                
                # Log the result
                self._log_detection(result, detector)
                
            except Exception as e:
                self.logger.error(f"Error in detector {detector.detector_type}: {str(e)}")
        
        return results
    
    def _apply_advanced_confidence(self, result: DetectionResult, conversation: Conversation) -> None:
        """Apply advanced confidence scoring to a detection result.
        
        Args:
            result: The detection result to enhance
            conversation: The conversation context
        """
        if not result.rut_detected:
            return
            
        detector_type = result.rut_type.value.lower()
        raw_confidence = result.confidence
        
        # Extract context factors from the conversation
        context_factors = self._extract_context_factors(conversation)
        
        # Apply advanced confidence calculation
        advanced_confidence = self.confidence_scorer.compute_advanced_confidence(
            detector_type=detector_type,
            raw_confidence=raw_confidence,
            signal_weights={detector_type: self.detector_weights.get(detector_type, 0.5)},
            context_factors=context_factors
        )
        
        # Update the detection result with advanced confidence info
        result.confidence = advanced_confidence["final"]
        result.evidence.update({
            "advanced_confidence": {
                "raw": advanced_confidence["raw"],
                "smoothed": advanced_confidence["smoothed"],
                "trend": advanced_confidence["trend"],
                "threshold": advanced_confidence["threshold"],
                "context_factors": context_factors
            }
        })
        
        # Update detection status based on advanced confidence
        result.rut_detected = advanced_confidence["is_detected"]
    
    def _extract_context_factors(self, conversation: Conversation) -> Dict[str, float]:
        """Extract contextual factors from the conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Dictionary of context factors and their values (-1 to 1)
        """
        factors = {}
        
        # Conversation length factor
        # Longer conversations might need more intervention
        msg_count = len(conversation.messages)
        length_factor = min(1.0, max(-1.0, (msg_count - 10) / 20))  # Scale around 10 messages
        factors["conversation_length"] = length_factor
        
        # Recent intervention factor
        # Recent interventions might reduce the need for new ones
        last_intervention_time = conversation.get_last_intervention_time()
        if last_intervention_time:
            messages_since = sum(1 for m in conversation.messages 
                              if m.timestamp > last_intervention_time)
            recency_factor = min(1.0, max(-1.0, (messages_since - 5) / 5))
            factors["intervention_recency"] = recency_factor
        
        # Message complexity factor
        # Complex messages might be more prone to issues
        if conversation.messages:
            recent = conversation.messages[-3:]
            avg_length = sum(len(m.content.split()) for m in recent) / max(1, len(recent))
            complexity_factor = min(1.0, max(-1.0, (avg_length - 50) / 100))
            factors["message_complexity"] = complexity_factor
            
        return factors
    
    def _log_detection(self, result: DetectionResult, detector: RutDetector) -> None:
        """Log a detection result.
        
        Args:
            result: The detection result to log
            detector: The detector that produced the result
        """
        if result.rut_detected:
            self.logger.info(
                f"Detector {detector.detector_type} found {result.rut_type.value} "
                f"with confidence {result.confidence:.2f}"
            )
        else:
            self.logger.debug(
                f"Detector {detector.detector_type} did not detect a rut "
                f"(confidence: {result.confidence:.2f})"
            )
    
    def detect_ruts(self, conversation: Conversation, window_size: Optional[int] = None) -> Optional[DetectionResult]:
        """Detect ruts in a conversation and return the highest confidence result.
        
        Args:
            conversation: The conversation to analyze.
            window_size: Optional size of the window to analyze.
                
        Returns:
            The highest confidence detection result if any rut was detected, None otherwise.
        """
        # Run all detectors
        results = self.analyze_conversation(conversation, window_size)
        
        # Filter to only positive detections
        positive_results = [r for r in results.values() if r.rut_detected]
        
        if not positive_results:
            self.logger.debug(f"No ruts detected in conversation {conversation.id}")
            return None
        
        # Get the highest confidence result
        highest_confidence = max(positive_results, key=lambda r: r.confidence)
        
        self.logger.info(
            f"Detected {highest_confidence.rut_type.value} rut in conversation {conversation.id} "
            f"with confidence {highest_confidence.confidence:.2f}"
        )
        
        return highest_confidence
    
    def add_analysis_to_conversation(self, conversation: Conversation, result: DetectionResult) -> None:
        """Add an analysis result to a conversation's history.
        
        Args:
            conversation: The conversation to add the result to.
            result: The detection result to add.
        """
        analysis_result = RutAnalysisResult(
            conversation_id=conversation.id,
            rut_detected=result.rut_detected,
            rut_type=result.rut_type,
            confidence=result.confidence,
            evidence=result.evidence
        )
        
        if not hasattr(conversation, 'rut_analyses'):
            conversation.rut_analyses = []
            
        conversation.rut_analyses.append(analysis_result)
        
        self.logger.debug(
            f"Added {result.rut_type.value} analysis result to conversation {conversation.id}"
        )
    
    def get_latest_analysis(self, conversation: Conversation) -> Optional[RutAnalysisResult]:
        """Get the most recent analysis result for a conversation.
        
        Args:
            conversation: The conversation to get the result for.
            
        Returns:
            The most recent analysis result, or None if no analyses exist.
        """
        if not hasattr(conversation, 'rut_analyses') or not conversation.rut_analyses:
            return None
        
        return conversation.rut_analyses[-1] 