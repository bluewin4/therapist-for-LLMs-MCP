"""
Tests for the intervention evaluator.

This module contains tests for the InterventionEvaluator class.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from mcp_therapist.core.interventions.evaluator import (
    InterventionEvaluator, 
    SuccessMetric, 
    EvaluationResult
)
from mcp_therapist.models.conversation import (
    Conversation,
    Message,
    MessageRole,
    InterventionPlan,
    RutType,
    InterventionStrategy
)
from mcp_therapist.core.detectors.base import DetectionResult


class TestInterventionEvaluator(unittest.TestCase):
    """Test cases for the InterventionEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock detector registry
        self.mock_registry = MagicMock()
        self.evaluator = InterventionEvaluator(detector_registry=self.mock_registry)
        
        # Create a test conversation with an intervention
        base_time = datetime.now()
        
        self.conversation = Conversation(
            id="test-convo",
            messages=[
                # Before intervention
                Message(
                    id="msg1",
                    role=MessageRole.USER,
                    content="I'm struggling with learning programming.",
                    timestamp=(base_time - timedelta(minutes=5)).timestamp()
                ),
                Message(
                    id="msg2",
                    role=MessageRole.ASSISTANT,
                    content="Learning to program can be challenging. What specific aspects are you struggling with?",
                    timestamp=(base_time - timedelta(minutes=4)).timestamp()
                ),
                Message(
                    id="msg3",
                    role=MessageRole.USER,
                    content="I just don't understand the concepts. It's too hard.",
                    timestamp=(base_time - timedelta(minutes=3)).timestamp()
                ),
                Message(
                    id="msg4",
                    role=MessageRole.ASSISTANT,
                    content="Programming does have a learning curve. Could you tell me which concepts you find difficult?",
                    timestamp=(base_time - timedelta(minutes=2)).timestamp()
                ),
                
                # Intervention
                Message(
                    id="intervention1",
                    role=MessageRole.SYSTEM,
                    content="[Self-reflection: I notice we're stuck in a cycle. Let me try a different approach.]",
                    timestamp=(base_time - timedelta(minutes=1)).timestamp(),
                    metadata={
                        "is_intervention": True,
                        "intervention_id": "int_test-convo_4",
                        "strategy_type": "REFRAMING",
                        "rut_type": "STAGNATION",
                        "confidence": 0.85,
                        "injection_method": "SELF_REFLECTION"
                    }
                ),
                
                # After intervention
                Message(
                    id="msg5",
                    role=MessageRole.ASSISTANT,
                    content="Instead of focusing on the difficulties, let's try a different approach. What if we start with a small, fun project? Would building a simple game interest you?",
                    timestamp=(base_time - timedelta(seconds=50)).timestamp()
                ),
                Message(
                    id="msg6",
                    role=MessageRole.USER,
                    content="That sounds interesting! What kind of game could I build as a beginner?",
                    timestamp=(base_time - timedelta(seconds=40)).timestamp()
                ),
                Message(
                    id="msg7",
                    role=MessageRole.ASSISTANT,
                    content="You could start with a number guessing game. It's simple but teaches important concepts like variables, conditionals, and loops.",
                    timestamp=(base_time - timedelta(seconds=30)).timestamp()
                ),
                Message(
                    id="msg8",
                    role=MessageRole.USER,
                    content="That sounds doable. How would I start?",
                    timestamp=(base_time - timedelta(seconds=20)).timestamp()
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
                "created_at": (base_time - timedelta(minutes=1)).isoformat()
            }
        )
        
        # Intervention ID
        self.intervention_id = "int_test-convo_4"
    
    def test_evaluate_intervention_success(self):
        """Test evaluation of a successful intervention."""
        # Mock the detector to indicate the rut is resolved
        mock_detector = MagicMock()
        mock_detector.analyze.return_value = DetectionResult(
            rut_detected=False,
            rut_type=RutType.STAGNATION,
            confidence=0.0,
            evidence={}
        )
        self.mock_registry.get_detector_for_rut_type.return_value = mock_detector
        
        # Evaluate the intervention
        result = self.evaluator.evaluate_intervention(
            self.conversation,
            self.intervention_id,
            self.intervention_plan
        )
        
        # Check the evaluation result
        self.assertTrue(result["success"])
        self.assertIn(SuccessMetric.RUT_RESOLUTION.value, result["metrics"])
        self.assertEqual(result["metrics"][SuccessMetric.RUT_RESOLUTION.value], 1.0)
        self.assertIn("success_score", result["analysis"])
    
    def test_evaluate_intervention_failure(self):
        """Test evaluation of a failed intervention."""
        # Mock the detector to indicate the rut still exists
        mock_detector = MagicMock()
        mock_detector.analyze.return_value = DetectionResult(
            rut_detected=True,
            rut_type=RutType.STAGNATION,
            confidence=0.9,  # Higher confidence than before
            evidence={"reason": "Conversation still focused on difficulties"}
        )
        self.mock_registry.get_detector_for_rut_type.return_value = mock_detector
        
        # Evaluate the intervention
        result = self.evaluator.evaluate_intervention(
            self.conversation,
            self.intervention_id,
            self.intervention_plan
        )
        
        # Check the evaluation result
        self.assertFalse(result["success"])
        self.assertIn(SuccessMetric.RUT_RESOLUTION.value, result["metrics"])
        self.assertEqual(result["metrics"][SuccessMetric.RUT_RESOLUTION.value], 0.0)
    
    def test_evaluate_topic_change(self):
        """Test evaluation of topic change after intervention."""
        # Create conversation with clear topic change
        base_time = datetime.now()
        
        conversation = Conversation(
            id="topic-change-convo",
            messages=[
                # Before intervention - topic: programming difficulties
                Message(
                    id="msg1",
                    role=MessageRole.USER,
                    content="Programming is hard. I don't understand variables and functions.",
                    timestamp=(base_time - timedelta(minutes=5)).timestamp()
                ),
                Message(
                    id="msg2",
                    role=MessageRole.ASSISTANT,
                    content="Variables can be confusing at first. What aspects of variables trouble you?",
                    timestamp=(base_time - timedelta(minutes=4)).timestamp()
                ),
                
                # Intervention
                Message(
                    id="intervention1",
                    role=MessageRole.SYSTEM,
                    content="[Self-reflection: We should try a different approach.]",
                    timestamp=(base_time - timedelta(minutes=3)).timestamp(),
                    metadata={
                        "is_intervention": True,
                        "intervention_id": "int_topic_2",
                        "strategy_type": "TOPIC_SWITCH",
                        "rut_type": "STAGNATION",
                        "confidence": 0.85,
                        "injection_method": "SELF_REFLECTION"
                    }
                ),
                
                # After intervention - topic: building projects
                Message(
                    id="msg3",
                    role=MessageRole.ASSISTANT,
                    content="Let's try a completely different approach. Have you considered learning through building projects?",
                    timestamp=(base_time - timedelta(minutes=2)).timestamp()
                ),
                Message(
                    id="msg4",
                    role=MessageRole.USER,
                    content="That might work better. What kind of projects?",
                    timestamp=(base_time - timedelta(minutes=1)).timestamp()
                )
            ]
        )
        
        # Mock the detector for basic resolution check
        mock_detector = MagicMock()
        mock_detector.analyze.return_value = DetectionResult(
            rut_detected=False,
            rut_type=RutType.STAGNATION,
            confidence=0.0,
            evidence={}
        )
        self.mock_registry.get_detector_for_rut_type.return_value = mock_detector
        
        # Evaluate the intervention
        result = self.evaluator.evaluate_intervention(
            conversation,
            "int_topic_2",
            self.intervention_plan
        )
        
        # Check the topic change score
        self.assertIn(SuccessMetric.TOPIC_CHANGE.value, result["metrics"])
        topic_change_score = result["metrics"][SuccessMetric.TOPIC_CHANGE.value]
        self.assertGreater(topic_change_score, 0.5)  # Should show significant topic change
    
    def test_evaluate_user_engagement(self):
        """Test evaluation of user engagement after intervention."""
        # Create conversation with improved user engagement
        base_time = datetime.now()
        
        conversation = Conversation(
            id="engagement-convo",
            messages=[
                # Before intervention - short, unenthusiastic responses
                Message(
                    id="msg1",
                    role=MessageRole.USER,
                    content="I don't know what to do.",
                    timestamp=(base_time - timedelta(minutes=5)).timestamp()
                ),
                Message(
                    id="msg2",
                    role=MessageRole.ASSISTANT,
                    content="Let's explore some options. What are you interested in learning?",
                    timestamp=(base_time - timedelta(minutes=4)).timestamp()
                ),
                Message(
                    id="msg3",
                    role=MessageRole.USER,
                    content="Not sure.",
                    timestamp=(base_time - timedelta(minutes=3)).timestamp()
                ),
                
                # Intervention
                Message(
                    id="intervention1",
                    role=MessageRole.SYSTEM,
                    content="[Self-reflection: The user seems disengaged. I should try to spark interest.]",
                    timestamp=(base_time - timedelta(minutes=2)).timestamp(),
                    metadata={
                        "is_intervention": True,
                        "intervention_id": "int_engage_3",
                        "strategy_type": "EXPLORATION",
                        "rut_type": "STAGNATION",
                        "confidence": 0.85,
                        "injection_method": "SELF_REFLECTION"
                    }
                ),
                
                # After intervention - longer, more enthusiastic responses
                Message(
                    id="msg4",
                    role=MessageRole.ASSISTANT,
                    content="Let me share something exciting! There's a new AI tool that lets complete beginners create amazing art with just a few words. Would you be interested in trying that?",
                    timestamp=(base_time - timedelta(minutes=1)).timestamp()
                ),
                Message(
                    id="msg5",
                    role=MessageRole.USER,
                    content="That actually sounds really cool! I've been wanting to create some art for my room but I'm not good at drawing. How does this AI art tool work? Can I try it for free?",
                    timestamp=base_time.timestamp()
                )
            ]
        )
        
        # Mock the detector for basic resolution check
        mock_detector = MagicMock()
        mock_detector.analyze.return_value = DetectionResult(
            rut_detected=False,
            rut_type=RutType.STAGNATION,
            confidence=0.0,
            evidence={}
        )
        self.mock_registry.get_detector_for_rut_type.return_value = mock_detector
        
        # Evaluate the intervention
        result = self.evaluator.evaluate_intervention(
            conversation,
            "int_engage_3",
            self.intervention_plan
        )
        
        # Check the user engagement score
        self.assertIn(SuccessMetric.USER_ENGAGEMENT.value, result["metrics"])
        engagement_score = result["metrics"][SuccessMetric.USER_ENGAGEMENT.value]
        self.assertGreater(engagement_score, 0.0)  # Should show improved engagement
    
    def test_evaluate_message_diversity(self):
        """Test evaluation of message diversity after intervention."""
        # Create conversation with improved message diversity
        base_time = datetime.now()
        
        conversation = Conversation(
            id="diversity-convo",
            messages=[
                # Before intervention - repetitive responses
                Message(
                    id="msg1",
                    role=MessageRole.USER,
                    content="How can I learn to program?",
                    timestamp=(base_time - timedelta(minutes=5)).timestamp()
                ),
                Message(
                    id="msg2",
                    role=MessageRole.ASSISTANT,
                    content="You can learn programming by taking online courses. There are many resources available.",
                    timestamp=(base_time - timedelta(minutes=4)).timestamp()
                ),
                Message(
                    id="msg3",
                    role=MessageRole.USER,
                    content="Which language should I start with?",
                    timestamp=(base_time - timedelta(minutes=3)).timestamp()
                ),
                Message(
                    id="msg4",
                    role=MessageRole.ASSISTANT,
                    content="You can start with Python. It's a beginner-friendly language. There are many resources available.",
                    timestamp=(base_time - timedelta(minutes=2)).timestamp()
                ),
                
                # Intervention
                Message(
                    id="intervention1",
                    role=MessageRole.SYSTEM,
                    content="[Self-reflection: I'm being repetitive. Let me provide more varied and specific information.]",
                    timestamp=(base_time - timedelta(minutes=1)).timestamp(),
                    metadata={
                        "is_intervention": True,
                        "intervention_id": "int_diverse_4",
                        "strategy_type": "REFLECTION",
                        "rut_type": "REPETITION",
                        "confidence": 0.85,
                        "injection_method": "SELF_REFLECTION"
                    }
                ),
                
                # After intervention - more diverse responses
                Message(
                    id="msg5",
                    role=MessageRole.ASSISTANT,
                    content="Python is excellent for beginners because of its readable syntax. I recommend starting with the official Python tutorial or interactive platforms like Codecademy. The key is consistent practice - try building small projects like a calculator or a simple game rather than just reading tutorials.",
                    timestamp=(base_time - timedelta(seconds=30)).timestamp()
                ),
                Message(
                    id="msg6",
                    role=MessageRole.USER,
                    content="What kind of projects would be good for a beginner?",
                    timestamp=(base_time - timedelta(seconds=20)).timestamp()
                ),
                Message(
                    id="msg7",
                    role=MessageRole.ASSISTANT,
                    content="For Python beginners, I suggest starting with: 1) A number guessing game to learn variables and conditionals, 2) A simple calculator to understand functions, 3) A to-do list app for file operations, and 4) A weather app using APIs. Each project builds on different skills and keeps learning engaging through practical application.",
                    timestamp=(base_time - timedelta(seconds=10)).timestamp()
                )
            ]
        )
        
        # Mock the detector for basic resolution check
        mock_detector = MagicMock()
        mock_detector.analyze.return_value = DetectionResult(
            rut_detected=False,
            rut_type=RutType.REPETITION,
            confidence=0.0,
            evidence={}
        )
        self.mock_registry.get_detector_for_rut_type.return_value = mock_detector
        
        # Evaluate the intervention
        result = self.evaluator.evaluate_intervention(
            conversation,
            "int_diverse_4",
            self.intervention_plan  # Using the same plan for simplicity
        )
        
        # Check the message diversity score
        self.assertIn(SuccessMetric.MESSAGE_DIVERSITY.value, result["metrics"])
        diversity_score = result["metrics"][SuccessMetric.MESSAGE_DIVERSITY.value]
        self.assertGreater(diversity_score, 0.0)  # Should show improved diversity
    
    def test_not_enough_messages_to_evaluate(self):
        """Test handling of conversations with insufficient messages after intervention."""
        # Create a conversation with only one message after intervention
        base_time = datetime.now()
        
        conversation = Conversation(
            id="short-convo",
            messages=[
                Message(
                    id="msg1",
                    role=MessageRole.USER,
                    content="Hello",
                    timestamp=(base_time - timedelta(minutes=2)).timestamp()
                ),
                # Intervention
                Message(
                    id="intervention1",
                    role=MessageRole.SYSTEM,
                    content="[Self-reflection: I should be more specific in my response.]",
                    timestamp=(base_time - timedelta(minutes=1)).timestamp(),
                    metadata={
                        "is_intervention": True,
                        "intervention_id": "int_short_1",
                        "strategy_type": "REFLECTION",
                        "rut_type": "STAGNATION",
                        "confidence": 0.85,
                        "injection_method": "SELF_REFLECTION"
                    }
                ),
                # Only one message after intervention
                Message(
                    id="msg2",
                    role=MessageRole.ASSISTANT,
                    content="Hi there! How can I assist you today?",
                    timestamp=base_time.timestamp()
                )
            ]
        )
        
        # Evaluate the intervention
        result = self.evaluator.evaluate_intervention(
            conversation,
            "int_short_1",
            self.intervention_plan
        )
        
        # Check that evaluation is pending
        self.assertIsNone(result["success"])
        self.assertEqual(result["metrics"], {})
        self.assertIn("status", result["analysis"])
        self.assertEqual(result["analysis"]["status"], "pending")
    
    def test_get_success_rate_by_rut_type(self):
        """Test gathering success rates grouped by rut type."""
        # Add multiple evaluation results
        self.evaluator.evaluation_results = {
            "convo1": [
                {
                    "intervention_id": "int1",
                    "success": True,
                    "analysis": {"rut_type": "STAGNATION"}
                },
                {
                    "intervention_id": "int2",
                    "success": False,
                    "analysis": {"rut_type": "STAGNATION"}
                }
            ],
            "convo2": [
                {
                    "intervention_id": "int3",
                    "success": True,
                    "analysis": {"rut_type": "REPETITION"}
                }
            ]
        }
        
        # Get success rates by rut type
        success_rates = self.evaluator.get_success_rate_by_rut_type()
        
        # Check the calculated rates
        self.assertIn("STAGNATION", success_rates)
        self.assertIn("REPETITION", success_rates)
        
        self.assertEqual(success_rates["STAGNATION"], 0.5)  # 1 out of 2
        self.assertEqual(success_rates["REPETITION"], 1.0)  # 1 out of 1
    
    def test_evaluation_result_structure(self):
        """Test the structure of evaluation results."""
        # Create a minimal evaluation result
        result = EvaluationResult(
            intervention_id="int-test",
            success=True,
            metrics={SuccessMetric.RUT_RESOLUTION.value: 1.0},
            analysis={"source": "test"}
        )
        
        # Check result structure
        self.assertIn("intervention_id", result)
        self.assertIn("success", result)
        self.assertIn("metrics", result)
        self.assertIn("analysis", result)
        self.assertIn("timestamp", result)
        
        # Check that result behaves like a dictionary
        self.assertEqual(result["intervention_id"], "int-test")
        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"][SuccessMetric.RUT_RESOLUTION.value], 1.0)


if __name__ == "__main__":
    unittest.main() 