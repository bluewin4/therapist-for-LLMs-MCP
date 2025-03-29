"""
Intervention Manager module.

This module provides the InterventionManager class, which serves as the main
coordinator for the intervention subsystem, integrating the strategist,
injector, and evaluator components.
"""

import uuid
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    InterventionPlan
)
from mcp_therapist.core.interventions.strategist import InterventionStrategist
from mcp_therapist.core.interventions.injector import InterventionInjector
from mcp_therapist.core.interventions.evaluator import InterventionEvaluator
from mcp_therapist.core.detectors.base import DetectionResult


logger = logging.getLogger("mcp_therapist")


class InterventionManager:
    """
    Manages the full intervention lifecycle, from detection to evaluation.
    
    The InterventionManager coordinates between three main components:
    - The InterventionStrategist: Analyzes conversations, detects ruts, and creates intervention plans
    - The InterventionInjector: Handles different methods of injecting interventions into conversations
    - The InterventionEvaluator: Evaluates the effectiveness of interventions after they've been applied
    
    Together, these components provide a complete system for detecting conversational ruts,
    planning and executing appropriate interventions, and learning from the results.
    """
    
    def __init__(
        self,
        strategist: Optional[InterventionStrategist] = None,
        injector: Optional[InterventionInjector] = None,
        evaluator: Optional[InterventionEvaluator] = None
    ):
        """
        Initialize the intervention manager with its component subsystems.
        
        Args:
            strategist: The strategist responsible for analyzing conversations and creating intervention plans
            injector: The injector responsible for applying interventions to conversations
            evaluator: The evaluator responsible for assessing intervention effectiveness
        """
        self.strategist = strategist or InterventionStrategist()
        self.injector = injector or InterventionInjector()
        self.evaluator = evaluator or InterventionEvaluator()
        self.intervention_history: Dict[str, Dict[str, Any]] = {}
    
    def analyze_conversation(self, conversation: Conversation) -> DetectionResult:
        """
        Analyze a conversation to detect potential ruts.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            A detection result indicating whether a rut was found
        """
        return self.strategist.analyze_conversation(conversation)
    
    def create_intervention_plan(
        self,
        conversation: Conversation,
        detection_result: DetectionResult
    ) -> InterventionPlan:
        """
        Create an intervention plan based on detection results.
        
        Args:
            conversation: The conversation context
            detection_result: The result from rut detection
            
        Returns:
            An intervention plan with strategy and parameters
        """
        return self.strategist.create_intervention_plan(conversation, detection_result)
    
    def craft_intervention_prompt(
        self,
        conversation: Conversation,
        intervention_plan: InterventionPlan
    ) -> str:
        """
        Craft the content for an intervention based on the plan.
        
        Args:
            conversation: The conversation context
            intervention_plan: The intervention plan to execute
            
        Returns:
            The intervention content as a string
        """
        return self.strategist.craft_intervention_prompt(conversation, intervention_plan)
    
    def inject_intervention(
        self,
        conversation: Conversation,
        intervention_content: str,
        intervention_plan: InterventionPlan,
        injection_method: Optional[str] = None
    ) -> Conversation:
        """
        Inject an intervention into the conversation using the specified method.
        
        Args:
            conversation: The conversation to inject the intervention into
            intervention_content: The content of the intervention
            intervention_plan: The plan for the intervention
            injection_method: The method to use for injection (if None, uses default)
            
        Returns:
            The updated conversation with the intervention
        """
        # Store the intervention plan in history for later evaluation
        intervention_id = f"int_{conversation.id}_{len(conversation.messages)}"
        self.intervention_history[intervention_id] = intervention_plan
        
        # Inject the intervention using the specified method
        return self.injector.inject_intervention(
            conversation=conversation,
            intervention_content=intervention_content,
            intervention_plan=intervention_plan,
            method=injection_method
        )
    
    def evaluate_intervention(
        self,
        conversation: Conversation,
        intervention_id: str,
        intervention_plan: InterventionPlan
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of an intervention.
        
        Args:
            conversation: The conversation containing the intervention
            intervention_id: The ID of the intervention to evaluate
            intervention_plan: The original plan for the intervention
            
        Returns:
            Evaluation results as a dictionary
        """
        return self.evaluator.evaluate_intervention(
            conversation,
            intervention_id,
            intervention_plan
        )
    
    def apply_pending_interventions(
        self,
        conversation: Conversation,
        message: Message
    ) -> Message:
        """
        Apply any pending interventions to a new message.
        
        This is used for PREPEND and INLINE injection methods where the
        intervention is added to a user's next message.
        
        Args:
            conversation: The conversation context
            message: The new message to potentially modify
            
        Returns:
            The message, possibly modified with interventions
        """
        return self.injector.apply_pending_interventions(conversation, message)
    
    def analyze_and_intervene(
        self,
        conversation: Conversation
    ) -> Tuple[Conversation, bool]:
        """
        Analyze a conversation and intervene if necessary.
        
        This is the main entry point for the intervention system, handling
        the full pipeline from detection to intervention.
        
        Args:
            conversation: The conversation to analyze and potentially intervene in
            
        Returns:
            A tuple of (updated_conversation, intervention_applied)
        """
        # Analyze the conversation for ruts
        detection_result = self.analyze_conversation(conversation)
        
        # If no rut detected or we shouldn't intervene, return early
        if not detection_result.rut_detected or not self.strategist.should_intervene(detection_result):
            return conversation, False
        
        # Create an intervention plan
        intervention_plan = self.create_intervention_plan(conversation, detection_result)
        
        # Craft the intervention content
        intervention_content = self.craft_intervention_prompt(conversation, intervention_plan)
        
        # Determine injection method based on strategy
        injection_method = None  # Use default from settings
        
        # Inject the intervention
        updated_conversation = self.inject_intervention(
            conversation,
            intervention_content,
            intervention_plan,
            injection_method
        )
        
        return updated_conversation, True
    
    def evaluate_intervention_effectiveness(
        self,
        conversation: Conversation
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the effectiveness of all interventions in a conversation.
        
        Args:
            conversation: The conversation containing interventions to evaluate
            
        Returns:
            A list of evaluation results for each intervention
        """
        results = []
        intervention_history = self.get_intervention_history()
        
        # Find all intervention messages
        for message in conversation.messages:
            metadata = message.metadata or {}
            if metadata.get("is_intervention"):
                intervention_id = metadata.get("intervention_id")
                if intervention_id and intervention_id in intervention_history:
                    # Evaluate the intervention
                    evaluation = self.evaluate_intervention(
                        conversation,
                        intervention_id,
                        intervention_history[intervention_id]
                    )
                    results.append(evaluation)
        
        return results
    
    def get_intervention_history(self) -> Dict[str, InterventionPlan]:
        """
        Get the history of interventions.
        
        Returns:
            A dictionary mapping intervention IDs to their plans
        """
        return self.intervention_history
    
    def get_intervention_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about intervention effectiveness.
        
        Returns:
            A dictionary containing statistics about interventions
        """
        # Get basic statistics from the evaluator
        evaluator_stats = self.evaluator.get_intervention_statistics()
        
        # Get injection method statistics from the injector
        injection_stats = self.injector.get_injection_method_statistics()
        
        # Combine the statistics
        combined_stats = evaluator_stats.copy()
        combined_stats["by_injection_method"] = injection_stats
        
        return combined_stats 