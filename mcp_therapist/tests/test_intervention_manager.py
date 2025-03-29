"""
Tests for the InterventionManager.

This module contains tests for the InterventionManager class, which coordinates
the detection, planning, and execution of interventions in the LLM Therapist system.
"""

import unittest
from unittest.mock import MagicMock, patch

from mcp_therapist.core.interventions.intervention_manager import InterventionManager
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType, RutAnalysisResult, InterventionPlan


class TestInterventionManager(unittest.TestCase):
    """Test cases for the InterventionManager."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the detector registry
        self.mock_detector_registry = MagicMock()
        self.mock_repetition_detector = MagicMock()
        self.mock_stagnation_detector = MagicMock()
        
        # Configure detector registry to return our mock detectors
        self.mock_detector_registry.get_detector.side_effect = lambda detector_type: {
            RutType.REPETITION: self.mock_repetition_detector,
            RutType.STAGNATION: self.mock_stagnation_detector
        }.get(detector_type)
        
        # Mock the strategist and prompt crafter
        self.mock_strategist = MagicMock()
        self.mock_prompt_crafter = MagicMock()
        
        # Create intervention manager with mocked components
        self.intervention_manager = InterventionManager(
            detector_registry=self.mock_detector_registry,
            strategist=self.mock_strategist,
            prompt_crafter=self.mock_prompt_crafter
        )
        
        # Create a test conversation
        self.test_conversation = MagicMock(spec=Conversation)
        self.test_conversation.id = "test_conversation"
        self.test_conversation.get_recent_messages.return_value = [
            Message(
                id="user_1",
                role=MessageRole.USER,
                content="Can you help me with Python?",
                timestamp=1000,
                metadata={}
            ),
            Message(
                id="assistant_1",
                role=MessageRole.ASSISTANT,
                content="I'd be happy to help with Python!",
                timestamp=1001,
                metadata={}
            )
        ]
    
    def test_no_rut_detected(self):
        """Test that no intervention is planned when no rut is detected."""
        # Configure detectors to not detect any ruts
        self.mock_repetition_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=False,
            rut_type=RutType.REPETITION,
            confidence=0.1,
            evidence={"reason": "No repetition found"}
        )
        
        self.mock_stagnation_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=False,
            rut_type=RutType.STAGNATION,
            confidence=0.2,
            evidence={"reason": "No stagnation found"}
        )
        
        # Run the intervention process
        result = self.intervention_manager.process_conversation(self.test_conversation)
        
        # Verify no intervention was planned
        self.assertIsNone(result)
        
        # Verify detectors were called
        self.mock_detector_registry.get_detector.assert_called()
        self.mock_repetition_detector.analyze.assert_called_once_with(self.test_conversation)
        self.mock_stagnation_detector.analyze.assert_called_once_with(self.test_conversation)
        
        # Verify strategist was not called
        self.mock_strategist.select_strategy.assert_not_called()
        
        # Verify prompt crafter was not called
        self.mock_prompt_crafter.craft_prompt.assert_not_called()
    
    def test_repetition_rut_detected(self):
        """Test intervention planning when repetition is detected."""
        # Configure repetition detector to detect a rut
        self.mock_repetition_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=True,
            rut_type=RutType.REPETITION,
            confidence=0.8,
            evidence={"phrase_repetition": ["Let me know if you need anything else"]}
        )
        
        self.mock_stagnation_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=False,
            rut_type=RutType.STAGNATION,
            confidence=0.2,
            evidence={"reason": "No stagnation found"}
        )
        
        # Configure strategist to return a strategy
        mock_intervention_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.REPETITION,
            strategy_type="REFLECTION",
            confidence=0.8,
            metadata={"rationale": "Repetition detected"}
        )
        self.mock_strategist.select_strategy.return_value = mock_intervention_plan
        
        # Configure prompt crafter to craft a prompt
        self.mock_prompt_crafter.craft_prompt.return_value = "I notice you've been saying 'Let me know if you need anything else' frequently. Let's try to be more specific in our assistance."
        
        # Run the intervention process
        result = self.intervention_manager.process_conversation(self.test_conversation)
        
        # Verify intervention was planned
        self.assertIsNotNone(result)
        self.assertEqual(result.conversation_id, "test_conversation")
        self.assertEqual(result.rut_type, RutType.REPETITION)
        self.assertEqual(result.strategy_type, "REFLECTION")
        
        # Verify detectors were called
        self.mock_detector_registry.get_detector.assert_called()
        self.mock_repetition_detector.analyze.assert_called_once_with(self.test_conversation)
        
        # Verify strategist was called with the correct rut result
        self.mock_strategist.select_strategy.assert_called_once()
        self.assertEqual(self.mock_strategist.select_strategy.call_args[0][0].rut_type, RutType.REPETITION)
        
        # Verify prompt crafter was called
        self.mock_prompt_crafter.craft_prompt.assert_called_once_with(mock_intervention_plan)
    
    def test_stagnation_rut_detected(self):
        """Test intervention planning when stagnation is detected."""
        # Configure stagnation detector to detect a rut
        self.mock_repetition_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=False,
            rut_type=RutType.REPETITION,
            confidence=0.2,
            evidence={"reason": "No repetition found"}
        )
        
        self.mock_stagnation_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=True,
            rut_type=RutType.STAGNATION,
            confidence=0.9,
            evidence={"topic_similarity": 0.95, "lack_of_progress": True}
        )
        
        # Configure strategist to return a strategy
        mock_intervention_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.STAGNATION,
            strategy_type="REFRAMING",
            confidence=0.9,
            metadata={"rationale": "Stagnation detected"}
        )
        self.mock_strategist.select_strategy.return_value = mock_intervention_plan
        
        # Configure prompt crafter to craft a prompt
        self.mock_prompt_crafter.craft_prompt.return_value = "I notice we might be going in circles. Let's approach this from a different angle."
        
        # Run the intervention process
        result = self.intervention_manager.process_conversation(self.test_conversation)
        
        # Verify intervention was planned
        self.assertIsNotNone(result)
        self.assertEqual(result.conversation_id, "test_conversation")
        self.assertEqual(result.rut_type, RutType.STAGNATION)
        self.assertEqual(result.strategy_type, "REFRAMING")
        
        # Verify detectors were called
        self.mock_detector_registry.get_detector.assert_called()
        self.mock_stagnation_detector.analyze.assert_called_once_with(self.test_conversation)
        
        # Verify strategist was called with the correct rut result
        self.mock_strategist.select_strategy.assert_called_once()
        self.assertEqual(self.mock_strategist.select_strategy.call_args[0][0].rut_type, RutType.STAGNATION)
        
        # Verify prompt crafter was called
        self.mock_prompt_crafter.craft_prompt.assert_called_once_with(mock_intervention_plan)
    
    def test_multiple_ruts_detected_highest_confidence_selected(self):
        """Test that highest confidence rut is selected when multiple ruts are detected."""
        # Configure both detectors to detect ruts
        self.mock_repetition_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=True,
            rut_type=RutType.REPETITION,
            confidence=0.7,
            evidence={"phrase_repetition": ["I'd be happy to help"]}
        )
        
        self.mock_stagnation_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=True,
            rut_type=RutType.STAGNATION,
            confidence=0.9,  # Higher confidence
            evidence={"topic_similarity": 0.95, "lack_of_progress": True}
        )
        
        # Configure strategist to return a strategy
        mock_intervention_plan = InterventionPlan(
            conversation_id="test_conversation",
            rut_type=RutType.STAGNATION,  # Should select stagnation due to higher confidence
            strategy_type="REFRAMING",
            confidence=0.9,
            metadata={"rationale": "Stagnation detected"}
        )
        self.mock_strategist.select_strategy.return_value = mock_intervention_plan
        
        # Run the intervention process
        result = self.intervention_manager.process_conversation(self.test_conversation)
        
        # Verify intervention was planned for stagnation (higher confidence)
        self.assertIsNotNone(result)
        self.assertEqual(result.rut_type, RutType.STAGNATION)
        
        # Verify strategist was called with the stagnation result
        self.mock_strategist.select_strategy.assert_called_once()
        self.assertEqual(self.mock_strategist.select_strategy.call_args[0][0].rut_type, RutType.STAGNATION)
    
    def test_cooldown_period_respected(self):
        """Test that cooldown period prevents consecutive interventions."""
        # Configure detector to detect a rut
        self.mock_repetition_detector.analyze.return_value = RutAnalysisResult(
            conversation_id="test_conversation",
            rut_detected=True,
            rut_type=RutType.REPETITION,
            confidence=0.8,
            evidence={"phrase_repetition": ["Let me know if you need anything else"]}
        )
        
        # Configure conversation to have a recent intervention
        self.test_conversation.get_last_intervention_time.return_value = 900  # 15 minutes ago
        
        # Set cooldown to 30 minutes
        with patch.object(self.intervention_manager, 'intervention_cooldown', 1800):
            # Run the intervention process
            result = self.intervention_manager.process_conversation(self.test_conversation)
            
            # Verify no intervention was planned due to cooldown
            self.assertIsNone(result)
            
            # Verify detector was still called
            self.mock_detector_registry.get_detector.assert_called()


if __name__ == "__main__":
    unittest.main() 