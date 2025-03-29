"""
Tests for the intervention injector.

This module contains tests for the InterventionInjector class.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

from mcp_therapist.core.interventions.injector import InterventionInjector, InjectionMethod
from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    InterventionPlan,
    RutType,
    InterventionStrategy
)


class TestInterventionInjector(unittest.TestCase):
    """Test cases for the InterventionInjector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.injector = InterventionInjector()
        
        # Create a test conversation
        self.conversation = Conversation(
            id="test-convo",
            messages=[
                Message(
                    id="msg1",
                    role=MessageRole.USER,
                    content="What can you tell me about programming?",
                    timestamp=datetime.now().timestamp()
                ),
                Message(
                    id="msg2",
                    role=MessageRole.ASSISTANT,
                    content="Programming is the process of creating instructions for computers.",
                    timestamp=datetime.now().timestamp()
                )
            ]
        )
        
        # Create a test intervention plan
        self.intervention_plan = InterventionPlan(
            conversation_id="test-convo",
            rut_type=RutType.STAGNATION,
            strategy_type=InterventionStrategy.REFRAMING.value,
            confidence=0.85,
            metadata={
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Test prompt
        self.prompt = "I notice we've been focusing only on the basics. Let's explore more advanced programming concepts."
    
    def test_inject_direct(self):
        """Test direct injection method."""
        # Inject the intervention
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt,
            InjectionMethod.DIRECT
        )
        
        # Check that a new message was added to the conversation
        self.assertEqual(len(updated_conv.messages), 3)
        
        # Check that the new message is an intervention with the correct content
        new_msg = updated_conv.messages[-1]
        self.assertEqual(new_msg.content, self.prompt)
        self.assertEqual(new_msg.role, MessageRole.ASSISTANT)
        self.assertTrue(new_msg.metadata.get("is_intervention", False))
        self.assertEqual(new_msg.metadata.get("strategy_type"), InterventionStrategy.REFRAMING.value)
        self.assertEqual(new_msg.metadata.get("rut_type"), RutType.STAGNATION.value)
        self.assertEqual(new_msg.metadata.get("injection_method"), InjectionMethod.DIRECT.value)
    
    def test_inject_self_reflection(self):
        """Test self-reflection injection method."""
        # Inject the intervention
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt,
            InjectionMethod.SELF_REFLECTION
        )
        
        # Check that a new message was added to the conversation
        self.assertEqual(len(updated_conv.messages), 3)
        
        # Check that the new message is a system message with the intervention as a reflection
        new_msg = updated_conv.messages[-1]
        self.assertTrue("[Self-reflection:" in new_msg.content)
        self.assertEqual(new_msg.role, MessageRole.SYSTEM)
        self.assertTrue(new_msg.metadata.get("is_intervention", False))
        self.assertEqual(new_msg.metadata.get("injection_method"), InjectionMethod.SELF_REFLECTION.value)
    
    def test_inject_prepend(self):
        """Test prepend injection method."""
        # Inject the intervention
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt,
            InjectionMethod.PREPEND
        )
        
        # Check that no new message was added
        self.assertEqual(len(updated_conv.messages), 2)
        
        # Check that a pending intervention was added to conversation metadata
        self.assertIn("pending_interventions", updated_conv.metadata)
        self.assertEqual(len(updated_conv.metadata["pending_interventions"]), 1)
        
        pending = updated_conv.metadata["pending_interventions"][0]
        self.assertEqual(pending["type"], "prepend")
        self.assertEqual(pending["content"], self.prompt)
    
    def test_inject_inline(self):
        """Test inline injection method."""
        # Inject the intervention
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt,
            InjectionMethod.INLINE
        )
        
        # Check that no new message was added
        self.assertEqual(len(updated_conv.messages), 2)
        
        # Check that a pending intervention was added to conversation metadata
        self.assertIn("pending_interventions", updated_conv.metadata)
        self.assertEqual(len(updated_conv.metadata["pending_interventions"]), 1)
        
        pending = updated_conv.metadata["pending_interventions"][0]
        self.assertEqual(pending["type"], "inline")
        self.assertEqual(pending["content"], self.prompt)
    
    def test_inject_metadata_only(self):
        """Test metadata-only injection method."""
        # Inject the intervention
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt,
            InjectionMethod.METADATA_ONLY
        )
        
        # Check that no new message was added
        self.assertEqual(len(updated_conv.messages), 2)
        
        # Check that the intervention was added to conversation metadata
        self.assertIn("interventions", updated_conv.metadata)
        self.assertEqual(len(updated_conv.metadata["interventions"]), 1)
        
        intervention = updated_conv.metadata["interventions"][0]
        self.assertEqual(intervention["prompt"], self.prompt)
        self.assertEqual(intervention["strategy_type"], InterventionStrategy.REFRAMING.value)
    
    def test_get_injection_method_stats(self):
        """Test retrieving injection method statistics."""
        # Inject two interventions with different methods
        self.injector.inject(self.conversation, self.intervention_plan, self.prompt, InjectionMethod.DIRECT)
        
        plan2 = InterventionPlan(
            conversation_id="test-convo",
            rut_type=RutType.REPETITION,
            strategy_type=InterventionStrategy.REFLECTION.value,
            confidence=0.75,
            metadata={"created_at": datetime.now().isoformat()}
        )
        self.injector.inject(self.conversation, plan2, "Another prompt", InjectionMethod.SELF_REFLECTION)
        
        # Update one intervention as successful
        for intervention in self.injector.intervention_history["test-convo"]:
            if intervention.get("method") == InjectionMethod.DIRECT.value:
                intervention["success"] = True
                break
        
        # Get the stats
        stats = self.injector.get_injection_method_stats()
        
        # Check that stats were calculated correctly
        self.assertIn(InjectionMethod.DIRECT.value, stats)
        self.assertIn(InjectionMethod.SELF_REFLECTION.value, stats)
        
        direct_stats = stats[InjectionMethod.DIRECT.value]
        self.assertEqual(direct_stats["total"], 1)
        self.assertEqual(direct_stats["successful"], 1)
        self.assertEqual(direct_stats["success_rate"], 1.0)
        
        reflection_stats = stats[InjectionMethod.SELF_REFLECTION.value]
        self.assertEqual(reflection_stats["total"], 1)
        self.assertEqual(reflection_stats["successful"], 0)
        self.assertEqual(reflection_stats["success_rate"], 0.0)
    
    def test_apply_pending_interventions_prepend(self):
        """Test applying a pending prepend intervention."""
        # Inject a pending prepend intervention
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt,
            InjectionMethod.PREPEND
        )
        
        # Create a new message to be modified
        message = Message(
            id="msg3",
            role=MessageRole.ASSISTANT,
            content="Python is a popular programming language.",
            timestamp=datetime.now().timestamp()
        )
        
        # Apply the pending intervention
        modified_msg = self.injector.apply_pending_interventions(updated_conv, message)
        
        # Check that the message was modified correctly
        self.assertTrue(self.prompt in modified_msg.content)
        self.assertIn("interventions", modified_msg.metadata)
        
        # Check that the pending interventions were cleared
        self.assertEqual(len(updated_conv.metadata["pending_interventions"]), 0)
    
    def test_apply_pending_interventions_inline(self):
        """Test applying a pending inline intervention."""
        # Inject a pending inline intervention
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt,
            InjectionMethod.INLINE
        )
        
        # Create a new message with multiple paragraphs to be modified
        message = Message(
            id="msg3",
            role=MessageRole.ASSISTANT,
            content="Python is a popular programming language.\n\nIt's known for its readability and versatility.",
            timestamp=datetime.now().timestamp()
        )
        
        # Apply the pending intervention
        modified_msg = self.injector.apply_pending_interventions(updated_conv, message)
        
        # Check that the message was modified correctly
        self.assertTrue(self.prompt in modified_msg.content)
        self.assertIn("interventions", modified_msg.metadata)
        
        # Check that the pending interventions were cleared
        self.assertEqual(len(updated_conv.metadata["pending_interventions"]), 0)
    
    def test_strategy_method_mapping(self):
        """Test that strategies are mapped to appropriate injection methods."""
        # Check that REFLECTION strategy is mapped to SELF_REFLECTION method
        self.assertEqual(
            self.injector.strategy_method_mapping[InterventionStrategy.REFLECTION],
            InjectionMethod.SELF_REFLECTION
        )
        
        # Check that HIGHLIGHT_INCONSISTENCY strategy is mapped to DIRECT method
        self.assertEqual(
            self.injector.strategy_method_mapping[InterventionStrategy.HIGHLIGHT_INCONSISTENCY],
            InjectionMethod.DIRECT
        )
    
    def test_default_method_selection(self):
        """Test that the default method is used when no method is specified."""
        # Temporarily change the default method
        self.injector.default_method = InjectionMethod.DIRECT.value
        
        # Inject with no method specified
        updated_conv = self.injector.inject(
            self.conversation, 
            self.intervention_plan, 
            self.prompt
        )
        
        # Check that the direct method was used (a new message was added)
        self.assertEqual(len(updated_conv.messages), 3)
        new_msg = updated_conv.messages[-1]
        self.assertEqual(new_msg.metadata.get("injection_method"), InjectionMethod.DIRECT.value)


if __name__ == "__main__":
    unittest.main() 