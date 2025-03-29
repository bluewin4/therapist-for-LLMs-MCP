"""
Tests for the TopicFixationDetector class.

This module contains tests for verifying the functionality of the
topic fixation detector, which identifies when conversations get
stuck on a specific topic.
"""

import unittest
from unittest.mock import MagicMock, patch
import uuid

from mcp_therapist.core.detectors.topic_fixation import TopicFixationDetector
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType


class TestTopicFixationDetector(unittest.TestCase):
    """Test cases for the TopicFixationDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a detector with test configuration
        self.detector = TopicFixationDetector(
            min_messages=3,
            window_size=3,
            similarity_threshold=0.5,
            confidence_threshold=0.3
        )
        
        # Create a test conversation
        self.conversation = self._create_test_conversation()
    
    def _create_test_conversation(self, messages=None):
        """Create a test conversation with sample messages."""
        conversation = Conversation(id=str(uuid.uuid4()))
        
        # Add default messages if none provided
        if messages is None:
            messages = [
                (MessageRole.USER, "Let's talk about Python programming."),
                (MessageRole.ASSISTANT, "Python is a great programming language for beginners and experts alike."),
                (MessageRole.USER, "What are some good Python libraries for data science?"),
                (MessageRole.ASSISTANT, "For data science in Python, popular libraries include Pandas, NumPy, and Scikit-learn.")
            ]
        
        # Add messages to the conversation
        for i, (role, content) in enumerate(messages):
            message = Message(
                id=f"msg_{i}",
                role=role,
                content=content
            )
            conversation.add_message(message)
        
        return conversation
    
    @patch('mcp_therapist.utils.topic.TopicAnalyzer.detect_topic_fixation')
    def test_no_topic_fixation(self, mock_detect_fixation):
        """Test that no topic fixation is detected when topics vary."""
        # Configure the mock to return no fixation
        mock_detect_fixation.return_value = (False, 0.2, None)
        
        # Test the detector
        detected, confidence, details = self.detector.detect(self.conversation)
        
        # Assert results
        self.assertFalse(detected)
        self.assertEqual(confidence, 0.2)
        self.assertIn("repeated_terms", details)
        mock_detect_fixation.assert_called_once()
    
    @patch('mcp_therapist.utils.topic.TopicAnalyzer.detect_topic_fixation')
    def test_topic_fixation_detected(self, mock_detect_fixation):
        """Test that topic fixation is detected when similar topics persist."""
        # Configure the mock to return fixation
        mock_detect_fixation.return_value = (
            True, 
            0.7, 
            ["python", "programming", "code"]
        )
        
        # Test the detector
        detected, confidence, details = self.detector.detect(self.conversation)
        
        # Assert results
        self.assertTrue(detected)
        self.assertEqual(confidence, 0.7)
        self.assertIn("repeated_terms", details)
        self.assertEqual(details["repeated_terms"], ["python", "programming", "code"])
        mock_detect_fixation.assert_called_once()
    
    def test_not_enough_messages(self):
        """Test that detector handles conversations with too few messages."""
        # Create a conversation with only 2 messages
        short_conversation = self._create_test_conversation([
            (MessageRole.USER, "Hello!"),
            (MessageRole.ASSISTANT, "Hi there!")
        ])
        
        # Test the detector
        detected, confidence, details = self.detector.detect(short_conversation)
        
        # Assert results
        self.assertFalse(detected)
        self.assertEqual(confidence, 0.0)
        self.assertIn("reason", details)
        self.assertEqual(details["reason"], "Not enough messages")
    
    @patch('mcp_therapist.utils.topic.TopicAnalyzer.detect_topic_fixation')
    def test_confidence_below_threshold(self, mock_detect_fixation):
        """Test that fixation with low confidence is not reported."""
        # Configure the mock to return fixation with low confidence
        mock_detect_fixation.return_value = (
            True,  # fixation detected
            0.2,   # but with low confidence
            ["python", "programming"]
        )
        
        # Test the detector
        detected, confidence, details = self.detector.detect(self.conversation)
        
        # Assert results
        self.assertFalse(detected)  # Should not report fixation due to low confidence
        self.assertEqual(confidence, 0.2)
        self.assertIn("repeated_terms", details)
        mock_detect_fixation.assert_called_once()
    
    def test_detector_type_and_rut_type(self):
        """Test that detector reports correct detector_type and rut_type."""
        self.assertEqual(self.detector.detector_type, "topic_fixation")
        self.assertEqual(self.detector.rut_type, RutType.TOPIC_FIXATION)
    
    @patch('mcp_therapist.utils.topic.TopicAnalyzer.detect_topic_fixation')
    def test_extract_message_texts(self, mock_detect_fixation):
        """Test that message texts are correctly extracted."""
        # Set up the mock to avoid issues
        mock_detect_fixation.return_value = (False, 0.0, None)
        
        # Create a conversation with some empty messages
        conversation = self._create_test_conversation([
            (MessageRole.USER, "Hello!"),
            (MessageRole.ASSISTANT, ""),  # Empty message
            (MessageRole.USER, "How are you?"),
            (MessageRole.ASSISTANT, "I'm doing well, thanks for asking!")
        ])
        
        # Extract message texts using the protected method
        texts = self.detector._extract_message_texts(conversation)
        
        # Assert results
        self.assertEqual(len(texts), 3)  # Should exclude the empty message
        self.assertEqual(texts[0], "Hello!")
        self.assertEqual(texts[1], "How are you?")
        self.assertEqual(texts[2], "I'm doing well, thanks for asking!")


if __name__ == "__main__":
    unittest.main() 