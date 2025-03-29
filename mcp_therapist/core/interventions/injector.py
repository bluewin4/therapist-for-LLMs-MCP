"""
Intervention injector for applying interventions to conversations.

This module provides an injector that applies selected interventions
to conversations using different injection methods.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from mcp_therapist.config.settings import settings
from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    InterventionPlan,
    InterventionStrategy
)
from mcp_therapist.utils.logging import logger


class InjectionMethod(str, Enum):
    """Methods for injecting interventions into conversations."""
    
    DIRECT = "DIRECT"  # Direct insertion as an assistant message
    SELF_REFLECTION = "SELF_REFLECTION"  # Add to a system message as a reflection
    PREPEND = "PREPEND"  # Prepend to the next assistant message
    INLINE = "INLINE"  # Include inline within the next assistant message
    METADATA_ONLY = "METADATA_ONLY"  # Only add to metadata, no visible change


class InterventionInjector:
    """Injector for applying interventions to conversations."""
    
    def __init__(self):
        """Initialize the intervention injector."""
        self.logger = logger
        
        # Configure defaults from settings
        self.default_method = getattr(
            settings, 
            "DEFAULT_INJECTION_METHOD", 
            InjectionMethod.SELF_REFLECTION.value
        )
        
        # Strategy-specific injection method mappings
        # Define which strategies use which injection methods by default
        self.strategy_method_mapping = {
            InterventionStrategy.REFLECTION: InjectionMethod.SELF_REFLECTION,
            InterventionStrategy.PROMPT_REFINEMENT: InjectionMethod.DIRECT,
            InterventionStrategy.METACOGNITIVE: InjectionMethod.SELF_REFLECTION,
            InterventionStrategy.HIGHLIGHT_INCONSISTENCY: InjectionMethod.DIRECT,
            # Default to self-reflection for other strategies
        }
        
        # Initialize success metrics tracking
        self.intervention_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def inject(
        self, 
        conversation: Conversation, 
        intervention_plan: InterventionPlan, 
        prompt: str,
        method: Optional[InjectionMethod] = None
    ) -> Conversation:
        """Inject an intervention into a conversation.
        
        Args:
            conversation: The conversation to inject the intervention into.
            intervention_plan: The intervention plan to execute.
            prompt: The crafted intervention prompt text.
            method: The injection method to use (optional).
            
        Returns:
            The updated conversation with the intervention injected.
        """
        # Determine the injection method to use
        if method is None:
            # Get method from strategy mapping or use default
            strategy = InterventionStrategy(intervention_plan.strategy_type)
            method = self.strategy_method_mapping.get(strategy, InjectionMethod(self.default_method))
        
        # Create intervention message metadata
        intervention_metadata = {
            "is_intervention": True,
            "intervention_id": f"int_{conversation.id}_{len(conversation.messages)}",
            "plan_id": id(intervention_plan),
            "strategy_type": intervention_plan.strategy_type,
            "rut_type": intervention_plan.rut_type.value,
            "confidence": intervention_plan.confidence,
            "injection_method": method.value
        }
        
        # Track intervention in history
        self._record_intervention(
            conversation.id, 
            intervention_plan, 
            method.value
        )
        
        # Apply the appropriate injection method
        if method == InjectionMethod.DIRECT:
            return self._inject_direct(conversation, prompt, intervention_metadata)
        elif method == InjectionMethod.SELF_REFLECTION:
            return self._inject_self_reflection(conversation, prompt, intervention_metadata)
        elif method == InjectionMethod.PREPEND:
            return self._inject_prepend(conversation, prompt, intervention_metadata)
        elif method == InjectionMethod.INLINE:
            return self._inject_inline(conversation, prompt, intervention_metadata)
        elif method == InjectionMethod.METADATA_ONLY:
            return self._inject_metadata_only(conversation, prompt, intervention_metadata)
        else:
            self.logger.warning(f"Unknown injection method: {method}. Using DIRECT instead.")
            return self._inject_direct(conversation, prompt, intervention_metadata)
    
    def _inject_direct(
        self, conversation: Conversation, prompt: str, metadata: Dict[str, Any]
    ) -> Conversation:
        """Inject an intervention as a direct assistant message.
        
        Args:
            conversation: The conversation to inject into.
            prompt: The intervention prompt.
            metadata: The intervention metadata.
            
        Returns:
            The updated conversation.
        """
        message = Message(
            id=f"msg_{conversation.id}_{len(conversation.messages)}",
            role=MessageRole.ASSISTANT,
            content=prompt,
            metadata=metadata
        )
        conversation.add_message(message)
        self.logger.info(f"Injected DIRECT intervention into conversation {conversation.id}")
        return conversation
    
    def _inject_self_reflection(
        self, conversation: Conversation, prompt: str, metadata: Dict[str, Any]
    ) -> Conversation:
        """Inject an intervention as a system message self-reflection.
        
        Args:
            conversation: The conversation to inject into.
            prompt: The intervention prompt.
            metadata: The intervention metadata.
            
        Returns:
            The updated conversation.
        """
        # Create a system message containing the self-reflection
        message = Message(
            id=f"msg_{conversation.id}_{len(conversation.messages)}",
            role=MessageRole.SYSTEM,
            content=f"[Self-reflection: {prompt}]",
            metadata=metadata
        )
        conversation.add_message(message)
        self.logger.info(f"Injected SELF_REFLECTION intervention into conversation {conversation.id}")
        return conversation
    
    def _inject_prepend(
        self, conversation: Conversation, prompt: str, metadata: Dict[str, Any]
    ) -> Conversation:
        """Flag the conversation for prepending the intervention to the next assistant message.
        
        This method adds metadata to the conversation indicating that the next
        assistant message should be prepended with the intervention.
        
        Args:
            conversation: The conversation to inject into.
            prompt: The intervention prompt.
            metadata: The intervention metadata.
            
        Returns:
            The updated conversation.
        """
        # Store the pending prepend in conversation metadata
        if "pending_interventions" not in conversation.metadata:
            conversation.metadata["pending_interventions"] = []
            
        pending = {
            "type": "prepend",
            "content": prompt,
            "metadata": metadata
        }
        conversation.metadata["pending_interventions"].append(pending)
        
        self.logger.info(f"Flagged conversation {conversation.id} for PREPEND intervention")
        return conversation
    
    def _inject_inline(
        self, conversation: Conversation, prompt: str, metadata: Dict[str, Any]
    ) -> Conversation:
        """Flag the conversation for including the intervention inline in the next assistant message.
        
        This method adds metadata to the conversation indicating that the next
        assistant message should include the intervention inline.
        
        Args:
            conversation: The conversation to inject into.
            prompt: The intervention prompt.
            metadata: The intervention metadata.
            
        Returns:
            The updated conversation.
        """
        # Store the pending inline injection in conversation metadata
        if "pending_interventions" not in conversation.metadata:
            conversation.metadata["pending_interventions"] = []
            
        pending = {
            "type": "inline",
            "content": prompt,
            "metadata": metadata
        }
        conversation.metadata["pending_interventions"].append(pending)
        
        self.logger.info(f"Flagged conversation {conversation.id} for INLINE intervention")
        return conversation
    
    def _inject_metadata_only(
        self, conversation: Conversation, prompt: str, metadata: Dict[str, Any]
    ) -> Conversation:
        """Add the intervention only to conversation metadata, without visible changes.
        
        Args:
            conversation: The conversation to inject into.
            prompt: The intervention prompt.
            metadata: The intervention metadata.
            
        Returns:
            The updated conversation.
        """
        # Store intervention in conversation metadata for awareness
        if "interventions" not in conversation.metadata:
            conversation.metadata["interventions"] = []
            
        metadata_entry = {
            "prompt": prompt,
            **metadata
        }
        conversation.metadata["interventions"].append(metadata_entry)
        
        self.logger.info(f"Added METADATA_ONLY intervention to conversation {conversation.id}")
        return conversation
    
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
        if "pending_interventions" not in conversation.metadata:
            return message
            
        pending = conversation.metadata["pending_interventions"]
        if not pending:
            return message
            
        modified_message = message
        
        for intervention in pending:
            if intervention["type"] == "prepend":
                # Prepend the intervention to the message
                modified_message.content = f"{intervention['content']}\n\n{modified_message.content}"
                
                # Mark the message as containing an intervention
                if "interventions" not in modified_message.metadata:
                    modified_message.metadata["interventions"] = []
                    
                modified_message.metadata["interventions"].append({
                    "type": "prepend",
                    **intervention["metadata"]
                })
                
            elif intervention["type"] == "inline":
                # For inline, we'll insert at a natural paragraph break if possible
                paragraphs = modified_message.content.split("\n\n")
                if len(paragraphs) > 1:
                    # Insert after the first paragraph
                    paragraphs.insert(1, f"\n\n{intervention['content']}\n\n")
                    modified_message.content = "".join(paragraphs)
                else:
                    # Just append if there are no paragraph breaks
                    modified_message.content = f"{modified_message.content}\n\n{intervention['content']}"
                
                # Mark the message as containing an intervention
                if "interventions" not in modified_message.metadata:
                    modified_message.metadata["interventions"] = []
                    
                modified_message.metadata["interventions"].append({
                    "type": "inline",
                    **intervention["metadata"]
                })
        
        # Clear the pending interventions
        conversation.metadata["pending_interventions"] = []
        
        return modified_message
    
    def _record_intervention(
        self, conversation_id: str, plan: InterventionPlan, method: str
    ) -> None:
        """Record an intervention in the history for later evaluation.
        
        Args:
            conversation_id: ID of the conversation.
            plan: The intervention plan.
            method: The injection method used.
        """
        if conversation_id not in self.intervention_history:
            self.intervention_history[conversation_id] = []
            
        self.intervention_history[conversation_id].append({
            "timestamp": plan.metadata.get("created_at"),
            "rut_type": plan.rut_type.value,
            "strategy_type": plan.strategy_type,
            "confidence": plan.confidence,
            "method": method,
            "plan_id": id(plan),
            "success": None  # To be evaluated later
        })
    
    def get_intervention_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get the intervention history for a conversation.
        
        Args:
            conversation_id: ID of the conversation.
            
        Returns:
            List of intervention records.
        """
        return self.intervention_history.get(conversation_id, [])
    
    def get_injection_method_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics about the effectiveness of different injection methods.
        
        Returns:
            Dictionary mapping method names to statistics.
        """
        stats: Dict[str, Dict[str, Union[int, float]]] = {}
        
        # Collect all interventions
        all_interventions = []
        for conv_interventions in self.intervention_history.values():
            all_interventions.extend(conv_interventions)
        
        if not all_interventions:
            return {}
            
        # Group by method
        for method in InjectionMethod:
            method_interventions = [i for i in all_interventions if i.get("method") == method.value]
            
            if not method_interventions:
                continue
                
            # Count total and successful interventions
            total = len(method_interventions)
            successful = sum(1 for i in method_interventions if i.get("success") is True)
            
            # Calculate success rate
            success_rate = successful / total if total > 0 else 0.0
            
            stats[method.value] = {
                "total": total,
                "successful": successful,
                "success_rate": success_rate
            }
        
        return stats 