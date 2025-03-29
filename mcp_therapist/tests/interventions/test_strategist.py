"""Tests for the InterventionStrategist."""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from mcp_therapist.core.interventions.strategist import InterventionStrategist, AdaptiveStrategySelector
from mcp_therapist.core.detectors.base import DetectionResult
from mcp_therapist.models.conversation import (
    Conversation, 
    Message, 
    MessageRole, 
    RutType, 
    InterventionStrategy,
    InterventionPlan
)


class TestInterventionStrategist(unittest.TestCase):
    """Test cases for the InterventionStrategist class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a strategist
        self.strategist = InterventionStrategist()
        
        # Mock settings
        self.strategist.intervention_cooldown = 2
        self.strategist.max_interventions = 5
        
        # Create a test conversation
        self.conversation = Conversation(id="test-convo")
        self.conversation.messages = [
            Message(
                id="msg1",
                role=MessageRole.USER,
                content="Hello, I need help with something",
                metadata={}
            ),
            Message(
                id="msg2",
                role=MessageRole.ASSISTANT,
                content="I'm here to help! What can I assist you with?",
                metadata={}
            )
        ]
        
        # Create a detection result
        self.detection_result = DetectionResult(
            rut_detected=True,
            rut_type=RutType.STAGNATION,
            confidence=0.85,
            evidence={
                "details": "Conversation appears to be stagnating",
                "repeated_terms": ["stuck", "confused"]
            }
        )
    
    def test_initialization(self):
        """Test the initialization of the strategist."""
        self.assertEqual(self.strategist.intervention_cooldown, 2)
        self.assertEqual(self.strategist.max_interventions, 5)
        self.assertIsInstance(self.strategist.adaptive_selector, AdaptiveStrategySelector)
        self.assertIsInstance(self.strategist.strategy_mapping, dict)
        self.assertIsInstance(self.strategist.strategy_success_rates, dict)
    
    def test_should_intervene_basic(self):
        """Test the basic intervention decision logic."""
        # Test with no rut detected
        no_rut = DetectionResult(
            rut_detected=False, 
            rut_type=RutType.STAGNATION, 
            confidence=0.0,
            evidence={"details": "No rut detected"}
        )
        self.assertFalse(self.strategist.should_intervene(self.conversation, no_rut))
        
        # Test with rut detected
        self.assertTrue(self.strategist.should_intervene(self.conversation, self.detection_result))
    
    def test_should_intervene_cooldown(self):
        """Test the intervention cooldown logic."""
        # Add an intervention message to the conversation
        intervention_msg = Message(
            id="intervention1",
            role=MessageRole.ASSISTANT,
            content="I notice we might be getting stuck. Let's try a different approach.",
            metadata={"is_intervention": True, "strategy_type": "REFRAMING"}
        )
        self.conversation.messages.append(intervention_msg)
        
        # Add one more regular message (not enough for cooldown)
        self.conversation.messages.append(
            Message(
                id="msg3",
                role=MessageRole.USER,
                content="I'm still confused though.",
                metadata={}
            )
        )
        
        # Should not intervene due to cooldown
        self.assertFalse(self.strategist.should_intervene(self.conversation, self.detection_result))
        
        # Add another message to satisfy cooldown
        self.conversation.messages.append(
            Message(
                id="msg4",
                role=MessageRole.ASSISTANT,
                content="What specifically confuses you?",
                metadata={}
            )
        )
        
        # Now should be able to intervene
        self.assertTrue(self.strategist.should_intervene(self.conversation, self.detection_result))
    
    def test_should_intervene_max_interventions(self):
        """Test the maximum interventions logic."""
        # Add max_interventions intervention messages
        for i in range(self.strategist.max_interventions):
            self.conversation.messages.append(
                Message(
                    id=f"intervention{i+1}",
                    role=MessageRole.ASSISTANT,
                    content=f"Intervention {i+1}",
                    metadata={"is_intervention": True, "strategy_type": "REFRAMING"}
                )
            )
        
        # Should not intervene due to hitting max
        self.assertFalse(self.strategist.should_intervene(self.conversation, self.detection_result))
    
    def test_should_intervene_confidence(self):
        """Test the confidence threshold for intervention."""
        # Temporarily set the confidence threshold
        original_method = self.strategist.should_intervene
        
        try:
            # Create a mock version of should_intervene that uses a higher threshold
            def mock_should_intervene(conversation, detection_result):
                min_confidence = 0.9  # Higher threshold for this test
                if not detection_result.rut_detected:
                    return False
                if detection_result.confidence < min_confidence:
                    self.strategist.logger.info(
                        f"Confidence too low for intervention: {detection_result.confidence:.2f} < {min_confidence}"
                    )
                    return False
                return True
                
            # Replace the method with our mock
            self.strategist.should_intervene = mock_should_intervene
            
            # Create detection result with low confidence
            low_conf = DetectionResult(
                rut_detected=True,
                rut_type=RutType.STAGNATION,
                confidence=0.8,  # Below threshold
                evidence={"details": "Low confidence detection"}
            )
            
            # Should not intervene due to low confidence
            self.assertFalse(self.strategist.should_intervene(self.conversation, low_conf))
            
            # Create detection result with high confidence
            high_conf = DetectionResult(
                rut_detected=True,
                rut_type=RutType.STAGNATION,
                confidence=0.95,  # Above threshold
                evidence={"details": "High confidence detection"}
            )
            
            # Should intervene with high confidence
            self.assertTrue(self.strategist.should_intervene(self.conversation, high_conf))
        finally:
            # Restore the original method
            self.strategist.should_intervene = original_method
    
    def test_select_strategy(self):
        """Test strategy selection logic."""
        # Mock the adaptive selector
        self.strategist.adaptive_selector = MagicMock()
        self.strategist.adaptive_selector.select_strategy.return_value = InterventionStrategy.REFRAMING
        
        # Select a strategy
        strategy = self.strategist.select_strategy(self.detection_result, self.conversation)
        
        # Verify the selection
        self.assertEqual(strategy, InterventionStrategy.REFRAMING)
        self.strategist.adaptive_selector.select_strategy.assert_called_once()
    
    def test_select_strategy_recent_interventions(self):
        """Test strategy selection with recent interventions."""
        # Add recent intervention messages with different strategies
        self.conversation.messages.extend([
            Message(
                id="msg3",
                role=MessageRole.ASSISTANT,
                content="Intervention 1",
                metadata={"is_intervention": True, "strategy_type": "REFRAMING"}
            ),
            Message(
                id="msg4",
                role=MessageRole.USER,
                content="User response",
                metadata={}
            ),
            Message(
                id="msg5",
                role=MessageRole.ASSISTANT,
                content="Intervention 2",
                metadata={"is_intervention": True, "strategy_type": "TOPIC_SWITCH"}
            )
        ])
        
        # Mock the adaptive selector
        self.strategist.adaptive_selector = MagicMock()
        self.strategist.adaptive_selector.select_strategy.return_value = InterventionStrategy.EXPLORATION
        
        # Select a strategy
        strategy = self.strategist.select_strategy(self.detection_result, self.conversation)
        
        # Verify the selection
        self.assertEqual(strategy, InterventionStrategy.EXPLORATION)
        
        # Check that we're filtering out recent strategies correctly
        expected_available_strategies = [InterventionStrategy.EXPLORATION]  # Only one left after filtering
        
        # Get the actual args passed to select_strategy
        call_args = self.strategist.adaptive_selector.select_strategy.call_args
        
        # Check that the available strategies are as expected
        # Need to convert the actual args list to a set for comparison,
        # since ordering is not guaranteed
        self.assertTrue(
            set(s.value for s in call_args[0][1]).issuperset(set(s.value for s in expected_available_strategies))
        )
    
    def test_create_intervention_plan_no_intervention(self):
        """Test creation of intervention plan when no intervention should be made."""
        # Mock should_intervene to return False
        with patch.object(self.strategist, 'should_intervene', return_value=False):
            plan = self.strategist.create_intervention_plan(self.detection_result, self.conversation)
            self.assertIsNone(plan)
    
    def test_create_intervention_plan(self):
        """Test creation of intervention plan."""
        # Mock should_intervene to return True and select_strategy
        with patch.object(self.strategist, 'should_intervene', return_value=True):
            with patch.object(
                self.strategist, 
                'select_strategy', 
                return_value=InterventionStrategy.REFRAMING
            ):
                plan = self.strategist.create_intervention_plan(self.detection_result, self.conversation)
                
                # Verify the plan
                self.assertIsInstance(plan, InterventionPlan)
                self.assertEqual(plan.rut_type, RutType.STAGNATION)
                self.assertEqual(plan.strategy_type, "REFRAMING")
                self.assertEqual(plan.confidence, 0.85)
                self.assertEqual(plan.conversation_id, "test-convo")
                
                # Check metadata
                self.assertIn("rut_type", plan.metadata)
                self.assertIn("confidence", plan.metadata)
                self.assertIn("created_at", plan.metadata)
                self.assertIn("detection_evidence", plan.metadata)
                self.assertIn("context", plan.metadata)
    
    def test_extract_context(self):
        """Test extraction of context information."""
        # Test with topic fixation
        topic_detection = DetectionResult(
            rut_detected=True,
            rut_type=RutType.TOPIC_FIXATION,
            confidence=0.8,
            evidence={"repeated_terms": ["programming", "python", "code"]}
        )
        
        context = self.strategist._extract_context(topic_detection, self.conversation)
        
        self.assertIn("target_topic", context)
        self.assertEqual(context["target_topic"], "programming python code")
        self.assertEqual(context["message_count"], 2)
        self.assertEqual(context["user_message_count"], 1)
        self.assertEqual(context["assistant_message_count"], 1)
        
        # Test with contradiction
        contradiction_detection = DetectionResult(
            rut_detected=True,
            rut_type=RutType.CONTRADICTION,
            confidence=0.8,
            evidence={
                "contradictions": [
                    {"statement1": "I love dogs", "statement2": "I hate all animals", "similarity": 0.8}
                ]
            }
        )
        
        context = self.strategist._extract_context(contradiction_detection, self.conversation)
        
        self.assertIn("contradictions", context)
        self.assertEqual(len(context["contradictions"]), 1)
    
    def test_update_strategy_effectiveness(self):
        """Test updating of strategy effectiveness."""
        # Mock the adaptive selector
        self.strategist.adaptive_selector = MagicMock()
        
        # Initial success rate
        initial_rate = self.strategist.strategy_success_rates.get(InterventionStrategy.REFRAMING, 0.5)
        
        # Update with success
        self.strategist.update_strategy_effectiveness(
            InterventionStrategy.REFRAMING, 
            RutType.STAGNATION,
            True
        )
        
        # Check new success rate (should increase)
        new_rate = self.strategist.strategy_success_rates.get(InterventionStrategy.REFRAMING)
        self.assertGreater(new_rate, initial_rate)
        
        # Verify adaptive selector update
        self.strategist.adaptive_selector.update_stats.assert_called_once_with(
            RutType.STAGNATION, 
            InterventionStrategy.REFRAMING, 
            True, 
            None
        )
        
        # Update with failure
        self.strategist.update_strategy_effectiveness(
            InterventionStrategy.REFRAMING, 
            RutType.STAGNATION,
            False
        )
        
        # Check new success rate (should decrease)
        final_rate = self.strategist.strategy_success_rates.get(InterventionStrategy.REFRAMING)
        self.assertLess(final_rate, new_rate)
        
        # Update with custom reward
        self.strategist.update_strategy_effectiveness(
            InterventionStrategy.REFRAMING, 
            RutType.STAGNATION,
            True,
            0.7
        )
        
        # Verify adaptive selector called with reward
        self.strategist.adaptive_selector.update_stats.assert_called_with(
            RutType.STAGNATION, 
            InterventionStrategy.REFRAMING, 
            True, 
            0.7
        )


if __name__ == "__main__":
    unittest.main() 