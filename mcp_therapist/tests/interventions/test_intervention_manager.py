"""
Tests for the intervention manager with injector and evaluator integration.

This module contains tests for the InterventionManager class with its
new components for Phase 4 implementation.
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime, timedelta

from mcp_therapist.core.interventions.manager import InterventionManager
from mcp_therapist.core.interventions.injector import InterventionInjector
from mcp_therapist.core.interventions.evaluator import InterventionEvaluator
from mcp_therapist.core.interventions.strategist import InterventionStrategist
from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    InterventionPlan,
    RutType,
    InterventionStrategy
)
from mcp_therapist.core.detectors.base import DetectionResult
from mcp_therapist.core.detectors import registry


class TestInterventionManager(unittest.TestCase):
    """Test cases for the InterventionManager class with injector and evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock components
        self.mock_strategist = MagicMock()
        self.mock_strategist.analyze_conversation = MagicMock()
        self.mock_strategist.should_intervene = MagicMock()
        self.mock_strategist.create_intervention_plan = MagicMock()
        self.mock_strategist.craft_intervention_prompt = MagicMock()
        
        self.mock_injector = MagicMock()
        self.mock_injector.inject_intervention = MagicMock()
        self.mock_injector.apply_pending_interventions = MagicMock()
        self.mock_injector.get_injection_method_statistics = MagicMock()
        
        self.mock_evaluator = MagicMock()
        self.mock_evaluator.evaluate_intervention = MagicMock()
        self.mock_evaluator.get_intervention_statistics = MagicMock()
        
        # Create the intervention manager with mock components
        self.manager = InterventionManager(
            strategist=self.mock_strategist,
            injector=self.mock_injector,
            evaluator=self.mock_evaluator
        )
        
        # Create a test conversation
        base_time = datetime.now()
        
        self.conversation = Conversation(
            id="test-convo-123",
            messages=[
                Message(
                    id="msg1",
                    role=MessageRole.USER,
                    content="I don't know what to do with my life.",
                    timestamp=(base_time - timedelta(minutes=5)).timestamp()
                ),
                Message(
                    id="msg2",
                    role=MessageRole.ASSISTANT,
                    content="That's a common feeling. What are some of your interests?",
                    timestamp=(base_time - timedelta(minutes=4)).timestamp()
                ),
                Message(
                    id="msg3",
                    role=MessageRole.USER,
                    content="I don't have any interests really.",
                    timestamp=(base_time - timedelta(minutes=3)).timestamp()
                ),
                Message(
                    id="msg4",
                    role=MessageRole.ASSISTANT,
                    content="Everyone has some interests. Think about what you enjoy doing in your free time.",
                    timestamp=(base_time - timedelta(minutes=2)).timestamp()
                ),
                Message(
                    id="msg5",
                    role=MessageRole.USER,
                    content="I don't enjoy anything. I'm just lost.",
                    timestamp=(base_time - timedelta(minutes=1)).timestamp()
                )
            ]
        )
        
        # Create a detection result
        self.detection_result = DetectionResult(
            rut_detected=True,
            rut_type=RutType.STAGNATION,
            confidence=0.85,
            evidence={
                "repeated_sentiment": "negative",
                "topic_fixation": "lack of direction",
                "message_count": 5
            }
        )
        
        # Create an intervention plan
        self.intervention_plan = InterventionPlan(
            conversation_id="test-convo-123",
            rut_type=RutType.STAGNATION,
            strategy_type=InterventionStrategy.REFRAMING.value,
            confidence=0.85,
            metadata={
                "created_at": datetime.now().isoformat(),
                "target_topics": ["lack of direction", "interests"]
            }
        )
        
        # Create intervention content
        self.intervention_content = (
            "I notice we might be stuck in a cycle. Let's try a different approach. "
            "Instead of focusing on what you don't enjoy, let's explore what you "
            "find meaningful, even if it's not something you'd call an 'interest'."
        )

    def test_analyze_conversation(self):
        """Test analyzing a conversation for interventions."""
        # Setup mock
        self.mock_strategist.analyze_conversation.return_value = self.detection_result
        
        # Call the method
        result = self.manager.analyze_conversation(self.conversation)
        
        # Verify
        self.mock_strategist.analyze_conversation.assert_called_once_with(self.conversation)
        self.assertEqual(result, self.detection_result)
    
    def test_create_intervention_plan(self):
        """Test creating an intervention plan."""
        # Setup mock
        self.mock_strategist.create_intervention_plan.return_value = self.intervention_plan
        
        # Call the method
        plan = self.manager.create_intervention_plan(self.conversation, self.detection_result)
        
        # Verify
        self.mock_strategist.create_intervention_plan.assert_called_once_with(
            self.conversation, 
            self.detection_result
        )
        self.assertEqual(plan, self.intervention_plan)
    
    def test_craft_intervention_prompt(self):
        """Test crafting an intervention prompt."""
        # Setup mock
        self.mock_strategist.craft_intervention_prompt.return_value = self.intervention_content
        
        # Call the method
        prompt = self.manager.craft_intervention_prompt(self.conversation, self.intervention_plan)
        
        # Verify
        self.mock_strategist.craft_intervention_prompt.assert_called_once_with(
            self.conversation, 
            self.intervention_plan
        )
        self.assertEqual(prompt, self.intervention_content)
    
    def test_inject_intervention_direct(self):
        """Test injecting an intervention directly."""
        # Setup
        intervention_id = "int_test-convo-123_5"
        
        # Mock injector to return an updated conversation
        updated_conversation = self.conversation.copy()
        new_message = Message(
            id=intervention_id,
            role=MessageRole.SYSTEM,
            content=self.intervention_content,
            metadata={"is_intervention": True}
        )
        updated_conversation.messages.append(new_message)
        
        self.mock_injector.inject_intervention.return_value = updated_conversation
        
        # Call the method
        result = self.manager.inject_intervention(
            self.conversation,
            self.intervention_content,
            self.intervention_plan,
            injection_method="DIRECT"
        )
        
        # Verify
        self.mock_injector.inject_intervention.assert_called_once_with(
            conversation=self.conversation,
            intervention_content=self.intervention_content,
            intervention_plan=self.intervention_plan,
            method="DIRECT"
        )
        self.assertEqual(result, updated_conversation)
    
    def test_evaluate_intervention(self):
        """Test evaluating an intervention."""
        # Setup
        intervention_id = "int_test-convo-123_5"
        evaluation_result = {
            "intervention_id": intervention_id,
            "success": True,
            "metrics": {"RUT_RESOLUTION": 1.0},
            "analysis": {"success_score": 0.85}
        }
        
        self.mock_evaluator.evaluate_intervention.return_value = evaluation_result
        
        # Call the method
        result = self.manager.evaluate_intervention(
            self.conversation,
            intervention_id,
            self.intervention_plan
        )
        
        # Verify
        self.mock_evaluator.evaluate_intervention.assert_called_once_with(
            self.conversation,
            intervention_id,
            self.intervention_plan
        )
        self.assertEqual(result, evaluation_result)
    
    def test_analyze_and_intervene_when_rut_detected(self):
        """Test the complete process of analyzing and intervening when a rut is detected."""
        # Setup mocks for the full process
        self.mock_strategist.analyze_conversation.return_value = self.detection_result
        self.mock_strategist.create_intervention_plan.return_value = self.intervention_plan
        self.mock_strategist.craft_intervention_prompt.return_value = self.intervention_content
        self.mock_strategist.should_intervene.return_value = True
        
        # Mock updated conversation after intervention
        updated_conversation = self.conversation.copy()
        intervention_id = f"int_{self.conversation.id}_{len(self.conversation.messages)}"
        new_message = Message(
            id=intervention_id,
            role=MessageRole.SYSTEM,
            content=self.intervention_content,
            metadata={
                "is_intervention": True,
                "intervention_id": intervention_id,
                "rut_type": RutType.STAGNATION.value,
                "strategy_type": InterventionStrategy.REFRAMING.value,
                "confidence": 0.85
            }
        )
        updated_conversation.messages.append(new_message)
        
        self.mock_injector.inject_intervention.return_value = updated_conversation
        
        # Call the method
        result = self.manager.analyze_and_intervene(self.conversation)
        
        # Verify the complete process was executed
        self.mock_strategist.analyze_conversation.assert_called_once_with(self.conversation)
        self.mock_strategist.should_intervene.assert_called_once_with(self.detection_result)
        self.mock_strategist.create_intervention_plan.assert_called_once_with(
            self.conversation, 
            self.detection_result
        )
        self.mock_strategist.craft_intervention_prompt.assert_called_once_with(
            self.conversation, 
            self.intervention_plan
        )
        self.mock_injector.inject_intervention.assert_called_once()
        self.assertEqual(result, (updated_conversation, True))
    
    def test_analyze_and_intervene_when_no_rut_detected(self):
        """Test the process when no rut is detected."""
        # Setup - no rut detected
        no_rut_detection = DetectionResult(
            rut_detected=False,
            rut_type=None,
            confidence=0.0,
            evidence={}
        )
        self.mock_strategist.analyze_conversation.return_value = no_rut_detection
        
        # Call the method
        result = self.manager.analyze_and_intervene(self.conversation)
        
        # Verify only analysis was done, no intervention
        self.mock_strategist.analyze_conversation.assert_called_once_with(self.conversation)
        self.mock_strategist.should_intervene.assert_not_called()
        self.mock_strategist.create_intervention_plan.assert_not_called()
        self.mock_injector.inject_intervention.assert_not_called()
        self.assertEqual(result, (self.conversation, False))
    
    def test_analyze_and_intervene_when_should_not_intervene(self):
        """Test the process when a rut is detected but we shouldn't intervene."""
        # Setup - rut detected but shouldn't intervene
        self.mock_strategist.analyze_conversation.return_value = self.detection_result
        self.mock_strategist.should_intervene.return_value = False
        
        # Call the method
        result = self.manager.analyze_and_intervene(self.conversation)
        
        # Verify analysis was done, but no intervention
        self.mock_strategist.analyze_conversation.assert_called_once_with(self.conversation)
        self.mock_strategist.should_intervene.assert_called_once_with(self.detection_result)
        self.mock_strategist.create_intervention_plan.assert_not_called()
        self.mock_injector.inject_intervention.assert_not_called()
        self.assertEqual(result, (self.conversation, False))
    
    def test_apply_pending_interventions(self):
        """Test applying pending interventions to a message."""
        # Setup
        message = Message(
            id="user_msg",
            role=MessageRole.USER,
            content="I still don't know what I want to do.",
            timestamp=datetime.now().timestamp()
        )
        
        # Mock injector
        modified_message = message.copy()
        modified_message.content = "I still don't know what I want to do. [But I'm willing to explore options]"
        self.mock_injector.apply_pending_interventions.return_value = modified_message
        
        # Call the method
        result = self.manager.apply_pending_interventions(self.conversation, message)
        
        # Verify
        self.mock_injector.apply_pending_interventions.assert_called_once_with(
            self.conversation, 
            message
        )
        self.assertEqual(result, modified_message)
    
    def test_get_intervention_statistics(self):
        """Test retrieving intervention statistics."""
        # Setup
        expected_stats = {
            "total_interventions": 10,
            "successful_interventions": 7,
            "success_rate": 0.7,
            "by_strategy": {
                "REFRAMING": {"total": 4, "successful": 3, "rate": 0.75},
                "EXPLORATION": {"total": 3, "successful": 2, "rate": 0.67},
                "REFLECTION": {"total": 3, "successful": 2, "rate": 0.67}
            },
            "by_rut_type": {
                "STAGNATION": {"total": 5, "successful": 4, "rate": 0.8},
                "REPETITION": {"total": 3, "successful": 2, "rate": 0.67},
                "NEGATIVITY": {"total": 2, "successful": 1, "rate": 0.5}
            },
            "by_injection_method": {
                "DIRECT": {"total": 3, "successful": 2, "rate": 0.67},
                "SELF_REFLECTION": {"total": 4, "successful": 3, "rate": 0.75},
                "PREPEND": {"total": 2, "successful": 1, "rate": 0.5},
                "INLINE": {"total": 1, "successful": 1, "rate": 1.0}
            }
        }
        
        self.mock_evaluator.get_intervention_statistics.return_value = {
            "total_interventions": 10,
            "successful_interventions": 7,
            "success_rate": 0.7,
            "by_strategy": {
                "REFRAMING": {"total": 4, "successful": 3, "rate": 0.75},
                "EXPLORATION": {"total": 3, "successful": 2, "rate": 0.67},
                "REFLECTION": {"total": 3, "successful": 2, "rate": 0.67}
            },
            "by_rut_type": {
                "STAGNATION": {"total": 5, "successful": 4, "rate": 0.8},
                "REPETITION": {"total": 3, "successful": 2, "rate": 0.67},
                "NEGATIVITY": {"total": 2, "successful": 1, "rate": 0.5}
            }
        }
        
        self.mock_injector.get_injection_method_statistics.return_value = {
            "DIRECT": {"total": 3, "successful": 2, "rate": 0.67},
            "SELF_REFLECTION": {"total": 4, "successful": 3, "rate": 0.75},
            "PREPEND": {"total": 2, "successful": 1, "rate": 0.5},
            "INLINE": {"total": 1, "successful": 1, "rate": 1.0}
        }
        
        # Call the method
        stats = self.manager.get_intervention_statistics()
        
        # Verify
        self.mock_evaluator.get_intervention_statistics.assert_called_once()
        self.mock_injector.get_injection_method_statistics.assert_called_once()
        
        # Check that the stats were combined correctly
        self.assertEqual(stats["total_interventions"], 10)
        self.assertEqual(stats["success_rate"], 0.7)
        self.assertEqual(stats["by_strategy"]["REFRAMING"]["rate"], 0.75)
        self.assertEqual(stats["by_rut_type"]["STAGNATION"]["rate"], 0.8)
        self.assertEqual(stats["by_injection_method"]["SELF_REFLECTION"]["rate"], 0.75)
    
    def test_evaluate_intervention_effectiveness(self):
        """Test evaluating the effectiveness of all interventions."""
        # Setup - conversation with multiple interventions
        conversation_with_interventions = self.conversation.copy()
        
        # Add intervention messages
        int1_id = "int_test-convo-123_5"
        int1 = Message(
            id=int1_id,
            role=MessageRole.SYSTEM,
            content="First intervention",
            timestamp=(datetime.now() - timedelta(minutes=10)).timestamp(),
            metadata={
                "is_intervention": True,
                "intervention_id": int1_id,
                "strategy_type": "REFRAMING",
                "rut_type": "STAGNATION",
                "confidence": 0.85
            }
        )
        
        int2_id = "int_test-convo-123_8"
        int2 = Message(
            id=int2_id,
            role=MessageRole.SYSTEM,
            content="Second intervention",
            timestamp=(datetime.now() - timedelta(minutes=5)).timestamp(),
            metadata={
                "is_intervention": True,
                "intervention_id": int2_id,
                "strategy_type": "EXPLORATION",
                "rut_type": "REPETITION",
                "confidence": 0.75
            }
        )
        
        conversation_with_interventions.messages.extend([int1, int2])
        
        # Mock evaluator
        int1_plan = InterventionPlan(
            conversation_id="test-convo-123",
            rut_type=RutType.STAGNATION,
            strategy_type=InterventionStrategy.REFRAMING.value,
            confidence=0.85
        )
        
        int2_plan = InterventionPlan(
            conversation_id="test-convo-123",
            rut_type=RutType.REPETITION,
            strategy_type=InterventionStrategy.EXPLORATION.value,
            confidence=0.75
        )
        
        # Mock getting the plans
        with patch.object(self.manager, 'get_intervention_history', return_value={
            int1_id: int1_plan,
            int2_id: int2_plan
        }):
            # Mock evaluation results
            eval1 = {"intervention_id": int1_id, "success": True, "metrics": {}}
            eval2 = {"intervention_id": int2_id, "success": False, "metrics": {}}
            
            self.mock_evaluator.evaluate_intervention.side_effect = [eval1, eval2]
            
            # Call the method
            results = self.manager.evaluate_intervention_effectiveness(conversation_with_interventions)
            
            # Verify
            self.assertEqual(len(results), 2)
            self.mock_evaluator.evaluate_intervention.assert_any_call(
                conversation_with_interventions, int1_id, int1_plan
            )
            self.mock_evaluator.evaluate_intervention.assert_any_call(
                conversation_with_interventions, int2_id, int2_plan
            )
            
            self.assertEqual(results[0], eval1)
            self.assertEqual(results[1], eval2)
    
    def test_integration_with_defaults(self):
        """Test that the manager initializes default components if none provided."""
        # Create a manager without providing components
        with patch('mcp_therapist.core.interventions.manager.InterventionStrategist') as mock_strat_class, \
             patch('mcp_therapist.core.interventions.manager.InterventionInjector') as mock_inj_class, \
             patch('mcp_therapist.core.interventions.manager.InterventionEvaluator') as mock_eval_class:
            
            # Configure mocks
            mock_strat_instance = MagicMock()
            mock_inj_instance = MagicMock()
            mock_eval_instance = MagicMock()
            
            mock_strat_class.return_value = mock_strat_instance
            mock_inj_class.return_value = mock_inj_instance
            mock_eval_class.return_value = mock_eval_instance
            
            # Create manager with defaults
            manager = InterventionManager()
            
            # Verify the default components were created
            self.assertEqual(manager.strategist, mock_strat_instance)
            self.assertEqual(manager.injector, mock_inj_instance)
            self.assertEqual(manager.evaluator, mock_eval_instance)
            
            # Check they were initialized
            mock_strat_class.assert_called_once()
            mock_inj_class.assert_called_once()
            mock_eval_class.assert_called_once()


if __name__ == "__main__":
    unittest.main() 