"""
Intervention evaluator for assessing intervention effectiveness.

This module provides an evaluator that assesses the effectiveness of
interventions based on various success metrics and post-intervention analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import math

from mcp_therapist.config.settings import settings
from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    InterventionPlan,
    RutType
)
from mcp_therapist.utils.logging import logger


class SuccessMetric(str, Enum):
    """Metrics for evaluating intervention success."""
    
    RUT_RESOLUTION = "RUT_RESOLUTION"  # Primary: Did the rut get resolved?
    CONFIDENCE_DECREASE = "CONFIDENCE_DECREASE"  # Did the confidence of the rut decrease?
    TOPIC_CHANGE = "TOPIC_CHANGE"  # Did the conversation topic change after intervention?
    USER_ENGAGEMENT = "USER_ENGAGEMENT"  # Did user engagement improve after intervention?
    USER_SENTIMENT = "USER_SENTIMENT"  # Did user sentiment improve after intervention?
    CONVERSATION_VELOCITY = "CONVERSATION_VELOCITY"  # Did the conversation pace improve?
    MESSAGE_DIVERSITY = "MESSAGE_DIVERSITY"  # Did message content diversity increase?


class EvaluationResult(dict):
    """Result of an intervention evaluation."""
    
    def __init__(
        self,
        intervention_id: str,
        success: bool,
        metrics: Dict[str, float],
        analysis: Dict[str, Any]
    ):
        """Initialize the evaluation result.
        
        Args:
            intervention_id: ID of the evaluated intervention.
            success: Overall success indicator.
            metrics: Dictionary of individual metric scores.
            analysis: Additional analysis details.
        """
        super().__init__({
            "intervention_id": intervention_id,
            "success": success,
            "metrics": metrics,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })


class InterventionEvaluator:
    """Evaluator for assessing intervention effectiveness."""
    
    def __init__(self, detector_registry=None):
        """Initialize the intervention evaluator.
        
        Args:
            detector_registry: Registry of rut detectors for re-analysis.
                If None, it will be imported when needed.
        """
        self.logger = logger
        self._detector_registry = detector_registry
        
        # Configure evaluation settings
        self.min_messages_to_evaluate = getattr(
            settings, "MIN_MESSAGES_TO_EVALUATE", 2
        )
        self.evaluation_window = getattr(
            settings, "EVALUATION_WINDOW", 5
        )
        
        # Define metric weights for overall success determination
        self.metric_weights = {
            SuccessMetric.RUT_RESOLUTION: 0.5,
            SuccessMetric.CONFIDENCE_DECREASE: 0.2,
            SuccessMetric.TOPIC_CHANGE: 0.1,
            SuccessMetric.USER_ENGAGEMENT: 0.1,
            SuccessMetric.USER_SENTIMENT: 0.05,
            SuccessMetric.CONVERSATION_VELOCITY: 0.025,
            SuccessMetric.MESSAGE_DIVERSITY: 0.025
        }
        
        # Store evaluation results
        self.evaluation_results: Dict[str, List[EvaluationResult]] = {}
    
    @property
    def detector_registry(self):
        """Get the detector registry, importing it if needed."""
        if self._detector_registry is None:
            from mcp_therapist.core.detectors.manager import DetectionManager
            self._detector_registry = DetectionManager()
        return self._detector_registry
    
    def evaluate_intervention(
        self, 
        conversation: Conversation,
        intervention_id: str,
        intervention_plan: InterventionPlan
    ) -> EvaluationResult:
        """Evaluate the effectiveness of an intervention.
        
        Args:
            conversation: The conversation containing the intervention.
            intervention_id: ID of the intervention to evaluate.
            intervention_plan: The original intervention plan.
            
        Returns:
            Evaluation result with success metrics.
        """
        # Find the intervention message
        intervention_index = -1
        intervention_message = None
        
        for i, msg in enumerate(conversation.messages):
            if msg.metadata.get("intervention_id") == intervention_id:
                intervention_index = i
                intervention_message = msg
                break
                
        if intervention_message is None:
            self.logger.warning(f"Intervention {intervention_id} not found in conversation")
            return EvaluationResult(
                intervention_id=intervention_id,
                success=False,
                metrics={},
                analysis={"error": "Intervention not found"}
            )
        
        # Check if there are enough messages after the intervention to evaluate
        messages_after = len(conversation.messages) - intervention_index - 1
        if messages_after < self.min_messages_to_evaluate:
            self.logger.info(
                f"Not enough messages ({messages_after}) after intervention to evaluate"
            )
            return EvaluationResult(
                intervention_id=intervention_id,
                success=None,  # Can't determine yet
                metrics={},
                analysis={"status": "pending", "messages_needed": self.min_messages_to_evaluate - messages_after}
            )
        
        # Get conversations slices before and after intervention
        before_slice = conversation.messages[
            max(0, intervention_index - self.evaluation_window):intervention_index
        ]
        after_slice = conversation.messages[
            intervention_index + 1:intervention_index + 1 + self.evaluation_window
        ]
        
        # Compute various metrics
        metrics = {}
        
        # Primary metric: Rut resolution
        # Re-analyze the conversation for the original rut type
        rut_resolved = self._evaluate_rut_resolution(
            conversation, intervention_plan.rut_type, intervention_index
        )
        metrics[SuccessMetric.RUT_RESOLUTION.value] = 1.0 if rut_resolved else 0.0
        
        # Secondary metrics
        metrics[SuccessMetric.CONFIDENCE_DECREASE.value] = self._evaluate_confidence_decrease(
            conversation, intervention_plan.rut_type, intervention_index
        )
        
        metrics[SuccessMetric.TOPIC_CHANGE.value] = self._evaluate_topic_change(
            before_slice, after_slice
        )
        
        metrics[SuccessMetric.USER_ENGAGEMENT.value] = self._evaluate_user_engagement(
            before_slice, after_slice
        )
        
        metrics[SuccessMetric.USER_SENTIMENT.value] = self._evaluate_sentiment_change(
            before_slice, after_slice
        )
        
        metrics[SuccessMetric.CONVERSATION_VELOCITY.value] = self._evaluate_conversation_velocity(
            before_slice, after_slice
        )
        
        metrics[SuccessMetric.MESSAGE_DIVERSITY.value] = self._evaluate_message_diversity(
            before_slice, after_slice
        )
        
        # Compute overall success score as weighted average of metrics
        success_score = 0.0
        total_weight = 0.0
        
        for metric, score in metrics.items():
            weight = self.metric_weights.get(SuccessMetric(metric), 0.0)
            success_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            success_score /= total_weight
        
        # Determine success based on threshold
        success_threshold = getattr(settings, "INTERVENTION_SUCCESS_THRESHOLD", 0.6)
        success = success_score >= success_threshold
        
        # Compile analysis details
        analysis = {
            "success_score": success_score,
            "success_threshold": success_threshold,
            "intervention_index": intervention_index,
            "messages_analyzed": len(before_slice) + len(after_slice),
            "rut_type": intervention_plan.rut_type.value,
            "strategy_type": intervention_plan.strategy_type,
            "method": intervention_message.metadata.get("injection_method", "unknown")
        }
        
        # Create the result
        result = EvaluationResult(
            intervention_id=intervention_id,
            success=success,
            metrics=metrics,
            analysis=analysis
        )
        
        # Store the result
        if conversation.id not in self.evaluation_results:
            self.evaluation_results[conversation.id] = []
        self.evaluation_results[conversation.id].append(result)
        
        self.logger.info(
            f"Evaluated intervention {intervention_id} in conversation {conversation.id}: "
            f"success={success}, score={success_score:.2f}"
        )
        
        return result
    
    def _evaluate_rut_resolution(
        self, conversation: Conversation, rut_type: RutType, intervention_index: int
    ) -> bool:
        """Evaluate whether the targeted rut was resolved after intervention.
        
        Args:
            conversation: The conversation to analyze.
            rut_type: The type of rut that was targeted.
            intervention_index: Index of the intervention message.
            
        Returns:
            True if the rut was resolved, False otherwise.
        """
        # Create a conversation slice after the intervention
        after_conversation = Conversation(
            id=f"{conversation.id}_eval",
            messages=conversation.messages[intervention_index + 1:],
            metadata=conversation.metadata.copy()
        )
        
        if not after_conversation.messages:
            return False
        
        # Analyze the conversation slice with the appropriate detector
        detector = self.detector_registry.get_detector_for_rut_type(rut_type)
        if detector is None:
            self.logger.warning(f"No detector found for rut type {rut_type}")
            return False
            
        detection_result = detector.analyze(after_conversation)
        
        # Rut is resolved if not detected in the post-intervention conversation
        return not detection_result.rut_detected
    
    def _evaluate_confidence_decrease(
        self, conversation: Conversation, rut_type: RutType, intervention_index: int
    ) -> float:
        """Evaluate whether the confidence in the rut decreased after intervention.
        
        Args:
            conversation: The conversation to analyze.
            rut_type: The type of rut that was targeted.
            intervention_index: Index of the intervention message.
            
        Returns:
            Score between 0.0 and 1.0 representing the degree of confidence decrease.
        """
        # Get the confidence before the intervention
        before_confidence = 0.0
        for i in range(intervention_index - 1, -1, -1):
            msg = conversation.messages[i]
            if "detection_result" in msg.metadata:
                detect_result = msg.metadata["detection_result"]
                if detect_result.get("rut_type") == rut_type.value:
                    before_confidence = detect_result.get("confidence", 0.0)
                    break
        
        if before_confidence == 0.0:
            # No prior detection found, use the intervention's confidence
            msg = conversation.messages[intervention_index]
            before_confidence = msg.metadata.get("confidence", 0.5)
        
        # Create a conversation slice after the intervention
        after_conversation = Conversation(
            id=f"{conversation.id}_eval",
            messages=conversation.messages[intervention_index + 1:],
            metadata=conversation.metadata.copy()
        )
        
        if not after_conversation.messages:
            return 0.0
        
        # Analyze the conversation slice with the appropriate detector
        detector = self.detector_registry.get_detector_for_rut_type(rut_type)
        if detector is None:
            self.logger.warning(f"No detector found for rut type {rut_type}")
            return 0.0
            
        detection_result = detector.analyze(after_conversation)
        after_confidence = detection_result.confidence
        
        # Calculate the decrease as a normalized value (0.0 to 1.0)
        if after_confidence >= before_confidence:
            return 0.0  # No decrease
            
        confidence_decrease = (before_confidence - after_confidence) / before_confidence
        return min(1.0, confidence_decrease)
    
    def _evaluate_topic_change(self, before_slice: List[Message], after_slice: List[Message]) -> float:
        """Evaluate whether the conversation topic changed after intervention.
        
        Args:
            before_slice: Messages before the intervention.
            after_slice: Messages after the intervention.
            
        Returns:
            Score between 0.0 and 1.0 representing the degree of topic change.
        """
        if not before_slice or not after_slice:
            return 0.0
            
        # For a basic implementation, use word overlap as a proxy for topic similarity
        before_text = " ".join([msg.content for msg in before_slice])
        after_text = " ".join([msg.content for msg in after_slice])
        
        # Get unique words (crude approximation of topics)
        before_words = set(before_text.lower().split())
        after_words = set(after_text.lower().split())
        
        # Calculate Jaccard similarity
        if not before_words or not after_words:
            return 0.0
            
        intersection = len(before_words.intersection(after_words))
        union = len(before_words.union(after_words))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Convert similarity to a change score (higher is more change)
        change_score = 1.0 - similarity
        
        return change_score
    
    def _evaluate_user_engagement(self, before_slice: List[Message], after_slice: List[Message]) -> float:
        """Evaluate whether user engagement improved after intervention.
        
        Args:
            before_slice: Messages before the intervention.
            after_slice: Messages after the intervention.
            
        Returns:
            Score between 0.0 and 1.0 representing the improvement in engagement.
        """
        if not before_slice or not after_slice:
            return 0.0
            
        # Get user messages from each slice
        before_user_msgs = [msg for msg in before_slice if msg.role == MessageRole.USER]
        after_user_msgs = [msg for msg in after_slice if msg.role == MessageRole.USER]
        
        if not before_user_msgs or not after_user_msgs:
            return 0.0
            
        # Use message length as a proxy for engagement
        avg_before_length = sum(len(msg.content) for msg in before_user_msgs) / len(before_user_msgs)
        avg_after_length = sum(len(msg.content) for msg in after_user_msgs) / len(after_user_msgs)
        
        # Calculate normalized change in message length
        if avg_before_length == 0:
            return 1.0 if avg_after_length > 0 else 0.0
            
        change_ratio = avg_after_length / avg_before_length
        
        # Normalize to 0.0-1.0 with a sigmoid-like function
        if change_ratio <= 1.0:
            # No improvement
            return 0.0
        else:
            # Map improvements to 0.0-1.0 with diminishing returns
            return min(1.0, (change_ratio - 1.0) / 2.0)
    
    def _evaluate_sentiment_change(self, before_slice: List[Message], after_slice: List[Message]) -> float:
        """Evaluate whether user sentiment improved after intervention.
        
        Args:
            before_slice: Messages before the intervention.
            after_slice: Messages after the intervention.
            
        Returns:
            Score between 0.0 and 1.0 representing sentiment improvement.
        """
        # For a basic implementation, we'll use a simplified sentiment analysis
        # In a real implementation, this would use a proper sentiment analyzer
        
        # Use a simple list of positive and negative words as a proxy
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "helpful", "useful", "interesting", "insightful", "clear", "thank",
            "thanks", "appreciate", "happy", "glad", "excited", "impressed"
        }
        
        negative_words = {
            "bad", "terrible", "poor", "awful", "unhelpful", "confusing",
            "unclear", "wrong", "incorrect", "disappointed", "frustrated",
            "angry", "sad", "confused", "error", "mistake", "not working"
        }
        
        def analyze_sentiment(messages):
            user_msgs = [msg.content.lower() for msg in messages if msg.role == MessageRole.USER]
            if not user_msgs:
                return 0.0  # Neutral
                
            combined_text = " ".join(user_msgs)
            words = combined_text.split()
            
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            total = pos_count + neg_count
            if total == 0:
                return 0.0  # Neutral
                
            return (pos_count - neg_count) / total  # Range: -1.0 to 1.0
        
        before_sentiment = analyze_sentiment(before_slice)
        after_sentiment = analyze_sentiment(after_slice)
        
        # Calculate the improvement, normalized to 0.0-1.0
        sentiment_change = after_sentiment - before_sentiment
        
        if sentiment_change <= 0:
            return 0.0  # No improvement
        else:
            # Normalize to 0.0-1.0
            return min(1.0, sentiment_change)
    
    def _evaluate_conversation_velocity(self, before_slice: List[Message], after_slice: List[Message]) -> float:
        """Evaluate whether conversation velocity improved after intervention.
        
        Args:
            before_slice: Messages before the intervention.
            after_slice: Messages after the intervention.
            
        Returns:
            Score between 0.0 and 1.0 representing velocity improvement.
        """
        if not before_slice or not after_slice:
            return 0.0
            
        # Use timestamps to calculate message frequency
        def calculate_velocity(messages):
            if len(messages) < 2:
                return 0.0
                
            timestamps = [msg.timestamp for msg in messages]
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0.0
            
            if avg_time_diff == 0.0:
                return 0.0  # Avoid division by zero
                
            return 1.0 / avg_time_diff  # Messages per second
        
        before_velocity = calculate_velocity(before_slice)
        after_velocity = calculate_velocity(after_slice)
        
        if before_velocity == 0.0 or after_velocity == 0.0:
            return 0.0
            
        # Calculate the relative improvement
        velocity_ratio = after_velocity / before_velocity
        
        if velocity_ratio <= 1.0:
            return 0.0  # No improvement
        else:
            # Normalize improvements to 0.0-1.0 with diminishing returns
            return min(1.0, (velocity_ratio - 1.0) / 2.0)
    
    def _evaluate_message_diversity(self, before_slice: List[Message], after_slice: List[Message]) -> float:
        """Evaluate whether message content diversity increased after intervention.
        
        Args:
            before_slice: Messages before the intervention.
            after_slice: Messages after the intervention.
            
        Returns:
            Score between 0.0 and 1.0 representing diversity improvement.
        """
        # This is a simplified diversity measurement using unique words ratio
        
        def calculate_diversity(messages):
            assistant_msgs = [msg.content.lower() for msg in messages if msg.role == MessageRole.ASSISTANT]
            if not assistant_msgs:
                return 0.0
                
            combined_text = " ".join(assistant_msgs)
            words = combined_text.split()
            
            if not words:
                return 0.0
                
            unique_words = set(words)
            
            return len(unique_words) / len(words)  # Unique word ratio
        
        before_diversity = calculate_diversity(before_slice)
        after_diversity = calculate_diversity(after_slice)
        
        # Calculate the improvement
        diversity_change = after_diversity - before_diversity
        
        if diversity_change <= 0:
            return 0.0  # No improvement
        else:
            # Normalize to 0.0-1.0
            return min(1.0, diversity_change * 5.0)  # Scale up small changes
    
    def get_evaluation_results(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get evaluation results for a specific conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            List of evaluation result dictionaries.
        """
        return self.evaluation_results.get(conversation_id, [])
    
    def get_success_rate_by_rut_type(self) -> Dict[str, float]:
        """Get success rates grouped by rut type.
        
        Returns:
            Dictionary mapping rut types to success rates.
        """
        success_counts = {}
        total_counts = {}
        
        # Collect all results
        all_results = []
        for results in self.evaluation_results.values():
            all_results.extend(results)
            
        # Count successes by rut type
        for result in all_results:
            rut_type = result["analysis"].get("rut_type")
            if not rut_type:
                continue
                
            if result["success"] is not None:  # Only count completed evaluations
                total_counts[rut_type] = total_counts.get(rut_type, 0) + 1
                if result["success"]:
                    success_counts[rut_type] = success_counts.get(rut_type, 0) + 1
        
        # Calculate success rates
        success_rates = {}
        for rut_type, total in total_counts.items():
            successes = success_counts.get(rut_type, 0)
            success_rates[rut_type] = successes / total if total > 0 else 0.0
            
        return success_rates
    
    def get_success_rate_by_strategy(self) -> Dict[str, float]:
        """Get success rates grouped by intervention strategy.
        
        Returns:
            Dictionary mapping strategies to success rates.
        """
        success_counts = {}
        total_counts = {}
        
        # Collect all results
        all_results = []
        for results in self.evaluation_results.values():
            all_results.extend(results)
            
        # Count successes by strategy
        for result in all_results:
            strategy = result["analysis"].get("strategy_type")
            if not strategy:
                continue
                
            if result["success"] is not None:  # Only count completed evaluations
                total_counts[strategy] = total_counts.get(strategy, 0) + 1
                if result["success"]:
                    success_counts[strategy] = success_counts.get(strategy, 0) + 1
        
        # Calculate success rates
        success_rates = {}
        for strategy, total in total_counts.items():
            successes = success_counts.get(strategy, 0)
            success_rates[strategy] = successes / total if total > 0 else 0.0
            
        return success_rates
    
    def get_success_rate_by_method(self) -> Dict[str, float]:
        """Get success rates grouped by injection method.
        
        Returns:
            Dictionary mapping injection methods to success rates.
        """
        success_counts = {}
        total_counts = {}
        
        # Collect all results
        all_results = []
        for results in self.evaluation_results.values():
            all_results.extend(results)
            
        # Count successes by method
        for result in all_results:
            method = result["analysis"].get("method")
            if not method:
                continue
                
            if result["success"] is not None:  # Only count completed evaluations
                total_counts[method] = total_counts.get(method, 0) + 1
                if result["success"]:
                    success_counts[method] = success_counts.get(method, 0) + 1
        
        # Calculate success rates
        success_rates = {}
        for method, total in total_counts.items():
            successes = success_counts.get(method, 0)
            success_rates[method] = successes / total if total > 0 else 0.0
            
        return success_rates 