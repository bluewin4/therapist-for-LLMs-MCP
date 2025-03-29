"""
Tests for the PromptCrafter.

This module contains tests for the PromptCrafter class, which generates
intervention prompts based on selected intervention strategies.
"""

import unittest
from unittest.mock import MagicMock, patch

from mcp_therapist.core.interventions.prompt_crafter import PromptCrafter
from mcp_therapist.models.conversation import InterventionPlan, RutType


class TestPromptCrafter(unittest.TestCase):
    """Test cases for the PromptCrafter."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a prompt crafter
        self.prompt_crafter = PromptCrafter()
        
        # Define some test intervention plans
        self.repetition_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.REPETITION,
            strategy_type="REFLECTION",
            confidence=0.8,
            metadata={
                "rationale": "Repeated phrases detected",
                "evidence": {
                    "phrase_repetition": ["Let me know if you need anything else"]
                }
            }
        )
        
        self.stagnation_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.STAGNATION,
            strategy_type="REFRAMING",
            confidence=0.9,
            metadata={
                "rationale": "Conversation is going in circles",
                "evidence": {
                    "topic_similarity": 0.95,
                    "lack_of_progress": True
                }
            }
        )
    
    def test_craft_repetition_reflection_prompt(self):
        """Test crafting a reflection prompt for repetition."""
        # Craft the prompt
        prompt = self.prompt_crafter.craft_prompt(self.repetition_plan)
        
        # Verify prompt content
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        
        # Check that the prompt contains reflection elements
        self.assertTrue(any(phrase in prompt.lower() for phrase in [
            "notice", "aware", "reflect", "repetition", "pattern"
        ]))
        
        # Check that the prompt contains the repeated phrase
        self.assertIn("Let me know if you need anything else", prompt)
    
    def test_craft_stagnation_reframing_prompt(self):
        """Test crafting a reframing prompt for stagnation."""
        # Craft the prompt
        prompt = self.prompt_crafter.craft_prompt(self.stagnation_plan)
        
        # Verify prompt content
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        
        # Check that the prompt contains reframing elements
        self.assertTrue(any(phrase in prompt.lower() for phrase in [
            "different approach", "perspective", "change", "reframe", "shift"
        ]))
    
    def test_unknown_strategy_fallback(self):
        """Test that unknown strategy types default to a generic intervention."""
        # Create a plan with an unknown strategy type
        unknown_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.REPETITION,
            strategy_type="UNKNOWN_STRATEGY",
            confidence=0.7,
            metadata={
                "rationale": "Need to try something different"
            }
        )
        
        # Craft the prompt
        prompt = self.prompt_crafter.craft_prompt(unknown_plan)
        
        # Verify prompt content - should get generic fallback
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
    
    def test_custom_template_insertion(self):
        """Test adding a custom template and using it."""
        # Add a custom template
        custom_strategy = "CUSTOM_TEST"
        custom_template = "This is a custom intervention template for {rut_type}."
        
        # Add the template to the crafter
        self.prompt_crafter.add_template(custom_strategy, custom_template)
        
        # Create a plan that uses this strategy
        custom_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.STAGNATION,
            strategy_type=custom_strategy,
            confidence=0.8,
            metadata={"rationale": "Testing custom template"}
        )
        
        # Craft the prompt
        prompt = self.prompt_crafter.craft_prompt(custom_plan)
        
        # Verify the custom template was used
        self.assertEqual(prompt, "This is a custom intervention template for STAGNATION.")
    
    def test_template_with_evidence(self):
        """Test template that incorporates evidence from the rut analysis."""
        # Create a plan with detailed evidence
        evidence_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.REPETITION,
            strategy_type="EVIDENCE_TEST",
            confidence=0.9,
            metadata={
                "rationale": "Testing evidence inclusion",
                "evidence": {
                    "repeated_phrases": ["I'd be happy to help", "Let me know if you need more"],
                    "repetition_count": 5
                }
            }
        )
        
        # Add a template that uses evidence
        evidence_template = "I notice you've said '{evidence[repeated_phrases][0]}' about {evidence[repetition_count]} times. Let's try something new."
        self.prompt_crafter.add_template("EVIDENCE_TEST", evidence_template)
        
        # Craft the prompt
        prompt = self.prompt_crafter.craft_prompt(evidence_plan)
        
        # Verify the evidence was incorporated
        self.assertEqual(prompt, "I notice you've said 'I'd be happy to help' about 5 times. Let's try something new.")
    
    def test_missing_template_variables(self):
        """Test graceful handling of missing template variables."""
        # Add a template with variables that may not be in all plans
        self.prompt_crafter.add_template(
            "MISSING_VAR_TEST", 
            "Testing with {metadata[missing_key]}, fallback: {metadata[missing_key]|default value}"
        )
        
        # Create a plan without those variables
        plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.STAGNATION,
            strategy_type="MISSING_VAR_TEST",
            confidence=0.7,
            metadata={"rationale": "Testing missing variables"}
        )
        
        # Craft the prompt
        prompt = self.prompt_crafter.craft_prompt(plan)
        
        # Verify fallback values are used
        self.assertEqual(prompt, "Testing with , fallback: default value")


if __name__ == "__main__":
    unittest.main() 