"""
Manager for handling the entire intervention process from detection to execution.

This module provides a manager that coordinates the detection, planning,
and execution of interventions in conversations.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from mcp_therapist.config.settings import settings
from mcp_therapist.core.detectors.manager import DetectionManager
from mcp_therapist.core.interventions.strategist import InterventionStrategist
from mcp_therapist.core.interventions.prompt_crafter import PromptCrafter
from mcp_therapist.core.interventions.injector import InterventionInjector, InjectionMethod
from mcp_therapist.core.interventions.evaluator import InterventionEvaluator
from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    InterventionPlan,
    InterventionStrategy
)
from mcp_therapist.utils.logging import logger


class InterventionManager:
    """Manager for coordinating the entire intervention process."""
    
    def __init__(self, detector_registry=None, strategist=None, prompt_crafter=None, 
                injector=None, evaluator=None):
        """Initialize the intervention manager.
        
        Args:
            detector_registry: Registry of rut detectors. If None, creates a DetectionManager.
            strategist: Intervention strategist. If None, creates an InterventionStrategist.
            prompt_crafter: Prompt crafter. If None, creates a PromptCrafter.
            injector: Intervention injector. If None, creates an InterventionInjector.
            evaluator: Intervention evaluator. If None, creates an InterventionEvaluator.
        """
        self.logger = logger
        
        # Create component instances or use provided ones
        self.detector_registry = detector_registry or DetectionManager()
        self.strategist = strategist or InterventionStrategist()
        self.prompt_crafter = prompt_crafter or PromptCrafter()
        self.injector = injector or InterventionInjector()
        self.evaluator = evaluator or InterventionEvaluator(detector_registry=self.detector_registry)
        
        # Set up cooldown parameters
        self.intervention_cooldown = getattr(settings, "INTERVENTION_COOLDOWN", 1800)  # Default 30 minutes
        
        self.logger.info("Intervention manager initialized")
    
    def analyze_conversation(self, conversation: Conversation) -> Tuple[bool, Dict[str, Any]]:
        """Analyze a conversation for ruts.
        
        Args:
            conversation: The conversation to analyze.
            
        Returns:
            A tuple containing (rut_detected, detection_results).
        """
        # Run the detection pipeline
        detection_result = self.detector_registry.analyze_conversation(conversation)
        
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "rut_detected": detection_result.rut_detected,
            "detection_result": detection_result.to_dict() if detection_result.rut_detected else None
        }
        
        return detection_result.rut_detected, analysis_result
    
    def create_intervention_plan(
        self, conversation: Conversation, detection_result
    ) -> Optional[InterventionPlan]:
        """Create an intervention plan for a detected rut.
        
        Args:
            conversation: The conversation to intervene in.
            detection_result: The result of the rut detection.
            
        Returns:
            An intervention plan, or None if no intervention should be made.
        """
        return self.strategist.create_intervention_plan(detection_result, conversation)
    
    def craft_intervention_prompt(
        self, plan: InterventionPlan, conversation: Conversation
    ) -> str:
        """Craft a prompt for an intervention plan.
        
        Args:
            plan: The intervention plan.
            conversation: The conversation to intervene in.
            
        Returns:
            The crafted prompt text.
        """
        return self.prompt_crafter.generate_prompt(plan, conversation)
    
    def inject_intervention(
        self, 
        conversation: Conversation, 
        intervention_plan: InterventionPlan, 
        prompt: str,
        method: Optional[InjectionMethod] = None
    ) -> Tuple[Conversation, str]:
        """Inject an intervention into a conversation.
        
        Args:
            conversation: The conversation to inject the intervention into.
            intervention_plan: The intervention plan.
            prompt: The crafted intervention prompt.
            method: The injection method to use (optional).
            
        Returns:
            A tuple containing (updated_conversation, intervention_id).
        """
        # Generate a unique ID for the intervention
        intervention_id = f"int_{conversation.id}_{len(conversation.messages)}"
        
        # Add intervention metadata to the plan for tracking
        intervention_plan.metadata["intervention_id"] = intervention_id
        intervention_plan.metadata["created_at"] = datetime.now().isoformat()
        
        # Inject the intervention using the specified method
        updated_conversation = self.injector.inject(
            conversation, intervention_plan, prompt, method
        )
        
        return updated_conversation, intervention_id
    
    def evaluate_intervention(
        self, conversation: Conversation, intervention_id: str, intervention_plan: InterventionPlan
    ) -> Dict[str, Any]:
        """Evaluate the effectiveness of an intervention.
        
        Args:
            conversation: The conversation containing the intervention.
            intervention_id: ID of the intervention to evaluate.
            intervention_plan: The original intervention plan.
            
        Returns:
            Evaluation result with success metrics.
        """
        return self.evaluator.evaluate_intervention(
            conversation, intervention_id, intervention_plan
        )
    
    def apply_pending_interventions(
        self, conversation: Conversation, message: Message
    ) -> Message:
        """Apply any pending interventions to an outgoing message.
        
        This should be called before sending any assistant message.
        
        Args:
            conversation: The conversation with pending interventions.
            message: The outgoing message to modify.
            
        Returns:
            The modified message with interventions applied.
        """
        return self.injector.apply_pending_interventions(conversation, message)
    
    def analyze_and_intervene(
        self, conversation: Conversation, window_size: Optional[int] = None
    ) -> Optional[Tuple[InterventionPlan, str]]:
        """Analyze a conversation and generate an intervention if needed.
        
        This is the main entry point that combines detection, planning, and
        intervention execution.
        
        Args:
            conversation: The conversation to analyze and potentially intervene in.
            window_size: Optional window size for analysis (messages to consider).
            
        Returns:
            A tuple containing (intervention_plan, prompt) if an intervention was made,
            or None if no intervention was needed.
        """
        # Analyze the conversation
        rut_detected, analysis = self.analyze_conversation(conversation)
        
        if not rut_detected:
            return None
            
        detection_result = self.detector_registry.create_detection_result_from_dict(
            analysis["detection_result"]
        )
        
        # Create an intervention plan
        intervention_plan = self.create_intervention_plan(conversation, detection_result)
        
        if intervention_plan is None:
            return None
            
        # Craft a prompt for the intervention
        prompt = self.craft_intervention_prompt(intervention_plan, conversation)
        
        # Determine injection method based on the intervention strategy
        strategy = InterventionStrategy(intervention_plan.strategy_type)
        method = self.injector.strategy_method_mapping.get(strategy, None)
        
        # Inject the intervention
        conversation, intervention_id = self.inject_intervention(
            conversation, intervention_plan, prompt, method
        )
        
        self.logger.info(
            f"Created intervention plan for conversation {conversation.id}: "
            f"{intervention_plan.strategy_type} - '{prompt[:50]}...'")
        
        return intervention_plan, prompt
    
    def evaluate_intervention_effectiveness(
        self, conversation: Conversation, intervention_plan: InterventionPlan, 
        detection_result
    ) -> bool:
        """Evaluate whether an intervention was effective.
        
        Args:
            conversation: The conversation to evaluate.
            intervention_plan: The intervention plan that was executed.
            detection_result: The post-intervention detection result.
            
        Returns:
            True if the intervention was effective, False otherwise.
        """
        # If no rut is detected now, the intervention was successful
        if not detection_result.rut_detected:
            success = True
        # If the same rut type is detected but with lower confidence, consider it partially successful
        elif detection_result.confidence < intervention_plan.confidence * 0.7:
            success = True
        # Otherwise, the intervention failed
        else:
            success = False
        
        # Update strategist with the success result if it has this method
        if hasattr(self.strategist, 'update_strategy_effectiveness'):
            strategy = InterventionStrategy(intervention_plan.strategy_type)
            self.strategist.update_strategy_effectiveness(
                strategy, intervention_plan.rut_type, success
            )
        
        # Find the intervention ID
        intervention_id = intervention_plan.metadata.get("intervention_id")
        if intervention_id:
            # Update intervention history in the injector
            for intervention in self.injector.intervention_history.get(conversation.id, []):
                if intervention.get("plan_id") == id(intervention_plan):
                    intervention["success"] = success
                    break
        
        self.logger.info(
            f"Intervention {intervention_plan.strategy_type} "
            f"{'succeeded' if success else 'failed'} for conversation {conversation.id}"
        )
        
        return success
    
    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Get statistics about intervention effectiveness.
        
        Returns:
            Dictionary of intervention statistics.
        """
        stats = {}
        
        # Get strategy success rates from strategist
        if hasattr(self.strategist, 'strategy_success_rates'):
            stats["strategy_success_rates"] = self.strategist.strategy_success_rates
        
        # Get injection method stats from injector
        method_stats = self.injector.get_injection_method_stats()
        if method_stats:
            stats["injection_method_stats"] = method_stats
        
        # Get evaluation stats by rut type
        rut_stats = self.evaluator.get_success_rate_by_rut_type()
        if rut_stats:
            stats["rut_type_success_rates"] = rut_stats
        
        # Get evaluation stats by strategy
        strategy_stats = self.evaluator.get_success_rate_by_strategy()
        if strategy_stats:
            stats["strategy_type_success_rates"] = strategy_stats
        
        return stats
    
    def get_intervention_history(self, conversation_id: str) -> Dict[str, Any]:
        """Get the intervention history for a conversation.
        
        Args:
            conversation_id: ID of the conversation.
            
        Returns:
            Dictionary containing intervention history data.
        """
        # Collect data from both injector and evaluator
        inj_history = self.injector.get_intervention_history(conversation_id)
        eval_history = self.evaluator.get_evaluation_results(conversation_id)
        
        return {
            "injections": inj_history,
            "evaluations": eval_history
        } 