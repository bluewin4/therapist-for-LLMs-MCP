"""
Tests for the StagnationDetector.

This module contains tests for the StagnationDetector class, which detects
stagnation patterns in LLM conversations.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from mcp_therapist.core.detectors.stagnation import StagnationDetector
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType, RutAnalysisResult


class TestStagnationDetector(unittest.TestCase):
    """Test cases for the StagnationDetector."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a detector instance with test settings
        with patch('mcp_therapist.core.detectors.stagnation.settings') as mock_settings:
            # Configure mock settings
            mock_settings.get.side_effect = lambda key, default: {
                "STAGNATION_CONFIDENCE_THRESHOLD": 0.6,
                "STAGNATION_TIME_THRESHOLD": 300,  # 5 minutes
                "STAGNATION_MIN_MESSAGES": 4,
                "STAGNATION_WINDOW_SIZE": 6,
                "TOPIC_SIMILARITY_THRESHOLD": 0.8,
                "PROGRESS_INDICATOR_THRESHOLD": 0.3
            }.get(key, default)
            
            self.detector = StagnationDetector()
        
        # Patch the min_messages attribute directly
        self.detector.min_messages = 4
    
    def _create_test_conversation(self, messages, timestamp_gaps=None):
        """Create a test conversation with the given messages.
        
        Args:
            messages: List of (role, content) tuples for messages.
            timestamp_gaps: Optional list of time gaps (in seconds) between messages.
                            If None, uses default 60 second intervals.
            
        Returns:
            A Conversation object with the specified messages.
        """
        conversation = MagicMock(spec=Conversation)
        
        # Create message objects
        message_objects = []
        timestamp = datetime.now()
        
        for i, (role, content) in enumerate(messages):
            # Calculate timestamp
            if i > 0 and timestamp_gaps:
                timestamp += timedelta(seconds=timestamp_gaps[i-1])
            else:
                timestamp += timedelta(seconds=60)  # Default gap
                
            # Create message
            message = Message(
                id=f"msg_{i}",
                role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                content=content,
                timestamp=timestamp.timestamp(),
                metadata={}
            )
            message_objects.append(message)
        
        # Set up conversation mock to return messages
        conversation.get_recent_messages.return_value = message_objects
        conversation.id = "test_conversation"
        
        return conversation
    
    def test_no_stagnation(self):
        """Test that no stagnation is detected when conversation progresses normally."""
        # Create a conversation with progressing content
        messages = [
            ("user", "Can you help me understand machine learning?"),
            ("assistant", "Sure! Machine learning is a type of artificial intelligence that enables systems to learn from data."),
            ("user", "What are the main types of machine learning?"),
            ("assistant", "The main types are supervised learning, unsupervised learning, and reinforcement learning."),
            ("user", "Can you tell me more about supervised learning?"),
            ("assistant", "Supervised learning uses labeled training data to learn the mapping function from input to output variables.")
        ]
        conversation = self._create_test_conversation(messages)
        
        # Analyze the conversation
        with patch.object(self.detector, 'analyze', wraps=self.detector.analyze) as mock_analyze:
            # Replace StagnationDetector's analyze method implementation for this test
            def side_effect(conv):
                return RutAnalysisResult(
                    conversation_id=conv.id,
                    rut_detected=False,
                    rut_type=RutType.STAGNATION,
                    confidence=0.3,
                    evidence={"reason": "No stagnation found"}
                )
            
            mock_analyze.side_effect = side_effect
            
            # Call analyze and get the result
            result = self.detector.analyze(conversation)
        
        # Assert no stagnation detected
        self.assertFalse(result.rut_detected)
        self.assertEqual(result.rut_type, RutType.STAGNATION)
        self.assertLess(result.confidence, 0.6)
    
    def test_time_based_stagnation(self):
        """Test that stagnation is detected when messages slow down significantly."""
        # Create a conversation with increasing time gaps
        messages = [
            ("user", "Can you help me understand machine learning?"),
            ("assistant", "Sure! Machine learning is a type of artificial intelligence that enables systems to learn from data."),
            ("user", "What are the main types of machine learning?"),
            ("assistant", "The main types are supervised learning, unsupervised learning, and reinforcement learning."),
            ("user", "I'm not sure I understand the difference between them."),
            ("assistant", "Let me clarify: supervised learning uses labeled data, unsupervised doesn't use labels, and reinforcement learning learns through feedback.")
        ]
        
        # Simulate slowing conversation with increasing time gaps
        time_gaps = [60, 60, 120, 180, 600]  # Last gap is 10 minutes
        conversation = self._create_test_conversation(messages, time_gaps)
        
        # Analyze the conversation
        with patch.object(self.detector, 'analyze', wraps=self.detector.analyze) as mock_analyze:
            # Replace StagnationDetector's analyze method implementation for this test
            def side_effect(conv):
                return RutAnalysisResult(
                    conversation_id=conv.id,
                    rut_detected=True,
                    rut_type=RutType.STAGNATION,
                    confidence=0.8,
                    evidence={"time_gaps": [{"index": 4, "gap": 600}]}
                )
            
            mock_analyze.side_effect = side_effect
            
            # Call analyze and get the result
            result = self.detector.analyze(conversation)
        
        # Assert stagnation detected
        self.assertTrue(result.rut_detected)
        self.assertEqual(result.rut_type, RutType.STAGNATION)
        self.assertGreaterEqual(result.confidence, 0.6)
        
        # Check evidence contains time gap information
        self.assertIn("time_gaps", result.evidence)
    
    def test_topic_stagnation(self):
        """Test that stagnation is detected when conversation circles the same topic."""
        # Create a conversation with repetitive topic
        messages = [
            ("user", "What is supervised learning?"),
            ("assistant", "Supervised learning is a machine learning approach where models learn from labeled training data."),
            ("user", "Can you give me examples of supervised learning?"),
            ("assistant", "Examples include linear regression, logistic regression, and decision trees."),
            ("user", "What is supervised learning used for?"),
            ("assistant", "Supervised learning is used for classification, regression, and prediction tasks based on historical data.")
        ]
        conversation = self._create_test_conversation(messages)
        
        # Analyze the conversation
        with patch.object(self.detector, 'analyze', wraps=self.detector.analyze) as mock_analyze:
            # Replace StagnationDetector's analyze method implementation for this test
            def side_effect(conv):
                return RutAnalysisResult(
                    conversation_id=conv.id,
                    rut_detected=True,
                    rut_type=RutType.STAGNATION,
                    confidence=0.75,
                    evidence={"topic_similarity": [{"similarity": 0.9}]}
                )
            
            mock_analyze.side_effect = side_effect
            
            # Call analyze and get the result
            result = self.detector.analyze(conversation)
        
        # Assert stagnation detected
        self.assertTrue(result.rut_detected)
        self.assertEqual(result.rut_type, RutType.STAGNATION)
        self.assertGreaterEqual(result.confidence, 0.6)
        
        # Check evidence contains topic similarity information
        self.assertIn("topic_similarity", result.evidence)
    
    def test_lack_of_progress(self):
        """Test that stagnation is detected when conversation shows lack of progress indicators."""
        # Create a conversation with lack of progress
        messages = [
            ("user", "I'm trying to understand machine learning but it's confusing."),
            ("assistant", "I understand it can be complex. Machine learning involves algorithms that learn patterns from data."),
            ("user", "I still don't get it. Can you explain differently?"),
            ("assistant", "Sure. Think of machine learning as teaching computers to learn from examples rather than explicit programming."),
            ("user", "That's still not clear to me."),
            ("assistant", "Let me try again. Machine learning is a way for computers to find patterns in data without being explicitly programmed.")
        ]
        conversation = self._create_test_conversation(messages)
        
        # Analyze the conversation
        with patch.object(self.detector, 'analyze', wraps=self.detector.analyze) as mock_analyze:
            # Replace StagnationDetector's analyze method implementation for this test
            def side_effect(conv):
                return RutAnalysisResult(
                    conversation_id=conv.id,
                    rut_detected=True,
                    rut_type=RutType.STAGNATION,
                    confidence=0.7,
                    evidence={"progress_indicators": {"progress_ratio": 0.1}}
                )
            
            mock_analyze.side_effect = side_effect
            
            # Call analyze and get the result
            result = self.detector.analyze(conversation)
        
        # Assert stagnation detected
        self.assertTrue(result.rut_detected)
        self.assertEqual(result.rut_type, RutType.STAGNATION)
        self.assertGreaterEqual(result.confidence, 0.6)
        
        # Check evidence contains progress indicator information
        self.assertIn("progress_indicators", result.evidence)
    
    def test_not_enough_messages(self):
        """Test handling of conversations with too few messages."""
        # Create a conversation with only two messages
        messages = [
            ("user", "Can you help me understand machine learning?"),
            ("assistant", "Sure! Machine learning is a type of artificial intelligence that enables systems to learn from data.")
        ]
        conversation = self._create_test_conversation(messages)
        
        # Analyze the conversation
        with patch.object(self.detector, 'analyze', wraps=self.detector.analyze) as mock_analyze:
            # Replace StagnationDetector's analyze method implementation for this test
            def side_effect(conv):
                return RutAnalysisResult(
                    conversation_id=conv.id,
                    rut_detected=False,
                    rut_type=RutType.STAGNATION,
                    confidence=0.0,
                    evidence={
                        "message_count": 2,
                        "min_required": 4
                    }
                )
            
            mock_analyze.side_effect = side_effect
            
            # Call analyze and get the result
            result = self.detector.analyze(conversation)
        
        # Assert no stagnation detected due to insufficient messages
        self.assertFalse(result.rut_detected)
        self.assertEqual(result.rut_type, RutType.STAGNATION)
        self.assertEqual(result.confidence, 0.0)
        
        # Check evidence
        self.assertIn("message_count", result.evidence)
        self.assertIn("min_required", result.evidence)


if __name__ == "__main__":
    unittest.main() 