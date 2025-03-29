"""
Tests for the NegativityDetector.

This module contains tests for the NegativityDetector class, which identifies
patterns of negative sentiment in conversations.
"""

import unittest
from unittest.mock import MagicMock, patch

from mcp_therapist.core.detectors.negativity import NegativityDetector
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType, RutAnalysisResult
from mcp_therapist.utils.sentiment import sentiment_analyzer


class TestNegativityDetector(unittest.TestCase):
    """Test cases for the NegativityDetector."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a detector instance with test settings
        with patch('mcp_therapist.core.detectors.negativity.settings') as mock_settings:
            # Configure mock settings
            mock_settings.get.side_effect = lambda key, default: {
                "NEGATIVITY_CONFIDENCE_THRESHOLD": 0.6,
                "NEGATIVITY_MIN_MESSAGES": 3,
                "NEGATIVITY_WINDOW_SIZE": 5,
                "CONSECUTIVE_NEG_THRESHOLD": 2,
                "OVERALL_NEG_RATIO_THRESHOLD": 0.6,
                "SENTIMENT_SHIFT_THRESHOLD": 0.3
            }.get(key, default)
            
            self.detector = NegativityDetector()
            
        # Patch the min_messages attribute directly
        self.detector.min_messages = 2
    
    def _create_test_conversation(self, assistant_messages, user_messages=None):
        """Create a test conversation with the given messages.
        
        Args:
            assistant_messages: List of strings for assistant messages.
            user_messages: List of strings for user messages. If None, generates default messages.
            
        Returns:
            A Conversation object with alternating user/assistant messages.
        """
        conversation = MagicMock(spec=Conversation)
        conversation.id = "test_conversation"
        
        # Create message objects
        messages = []
        
        # Generate default user messages if not provided
        if user_messages is None:
            user_messages = [f"User message {i}" for i in range(len(assistant_messages))]
        
        # Ensure lists are the same length
        while len(user_messages) < len(assistant_messages):
            user_messages.append(f"User message {len(user_messages)}")
        
        # Create alternating messages
        for i in range(len(assistant_messages)):
            # Add a user message
            user_msg = Message(
                id=f"user_{i}",
                role=MessageRole.USER,
                content=user_messages[i],
                timestamp=i*2,
                metadata={}
            )
            messages.append(user_msg)
            
            # Add the assistant message
            assistant_msg = Message(
                id=f"assistant_{i}",
                role=MessageRole.ASSISTANT,
                content=assistant_messages[i],
                timestamp=i*2+1,
                metadata={}
            )
            messages.append(assistant_msg)
        
        # Set up conversation mock to return messages
        conversation.get_recent_messages.return_value = messages
        
        return conversation
    
    def test_no_negativity(self):
        """Test that no negativity is detected in neutral/positive conversation."""
        # Mock the sentiment analyzer
        with patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_sentiment') as mock_analyze, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_messages') as mock_analyze_msgs, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.detect_sentiment_shift') as mock_detect_shift:
            
            # Configure sentiment analysis mocks
            mock_analyze.return_value = {
                "sentiment": "positive",
                "score": 0.8,
                "negative": False,
                "positive": True
            }
            
            mock_analyze_msgs.return_value = {
                "overall_sentiment": "positive",
                "average_score": 0.75,
                "negative_count": 0,
                "positive_count": 3,
                "neutral_count": 0,
                "negative_ratio": 0.0,
                "positive_ratio": 1.0
            }
            
            mock_detect_shift.return_value = {
                "sentiment_shift": False,
                "shift_magnitude": 0.1,
                "shift_direction": "positive"
            }
            
            # Create a conversation with positive assistant messages
            assistant_messages = [
                "I'm happy to help you with your question!",
                "That's a great point. Let me elaborate further.",
                "You're making excellent progress in understanding this concept."
            ]
            
            conversation = self._create_test_conversation(assistant_messages)
            
            # Test detector
            result = self.detector.analyze(conversation)
            
            # Verify no negativity detected
            self.assertFalse(result.rut_detected)
            self.assertEqual(result.rut_type, RutType.NEGATIVITY)
            self.assertLess(result.confidence, 0.6)
    
    def test_consecutive_negative_messages(self):
        """Test detection of consecutive negative messages."""
        # Mock the sentiment analyzer
        with patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_sentiment') as mock_analyze, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_messages') as mock_analyze_msgs, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.detect_sentiment_shift') as mock_detect_shift:
            
            # Configure sentiment analysis mocks
            def sentiment_side_effect(text):
                if "sorry" in text.lower() or "apologize" in text.lower() or "can't" in text.lower():
                    return {
                        "sentiment": "negative",
                        "score": 0.3,
                        "negative": True,
                        "positive": False
                    }
                return {
                    "sentiment": "neutral",
                    "score": 0.5,
                    "negative": False,
                    "positive": False
                }
            
            mock_analyze.side_effect = sentiment_side_effect
            
            mock_analyze_msgs.return_value = {
                "overall_sentiment": "negative",
                "average_score": 0.35,
                "negative_count": 2,
                "positive_count": 0,
                "neutral_count": 1,
                "negative_ratio": 0.67,
                "positive_ratio": 0.0
            }
            
            mock_detect_shift.return_value = {
                "sentiment_shift": False,
                "shift_magnitude": 0.15,
                "shift_direction": "negative"
            }
            
            # Create a conversation with negative assistant messages
            assistant_messages = [
                "I understand your question.",
                "I'm sorry, but I can't help with that specific request.",
                "I apologize, but I'm unable to process that kind of information."
            ]
            
            conversation = self._create_test_conversation(assistant_messages)
            
            # Test detector
            result = self.detector.analyze(conversation)
            
            # Verify negativity detected
            self.assertTrue(result.rut_detected)
            self.assertEqual(result.rut_type, RutType.NEGATIVITY)
            self.assertGreaterEqual(result.confidence, 0.6)
            
            # Check evidence contains consecutive negativity information
            self.assertIn("consecutive_negatives", result.evidence)
            self.assertTrue(result.evidence["consecutive_negatives"]["detected"])
    
    def test_sentiment_mismatch(self):
        """Test detection of sentiment mismatch between user and assistant."""
        # Mock the sentiment analyzer
        with patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_sentiment') as mock_analyze, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_messages') as mock_analyze_msgs, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.detect_sentiment_shift') as mock_detect_shift:
            
            # Configure sentiment analysis mocks - assistant negative, user positive
            def analyze_messages_side_effect(texts):
                if "Great news" in str(texts) or "excited" in str(texts):
                    return {
                        "overall_sentiment": "positive",
                        "average_score": 0.8,
                        "negative_count": 0,
                        "positive_count": 3,
                        "neutral_count": 0,
                        "negative_ratio": 0.0,
                        "positive_ratio": 1.0
                    }
                else:
                    return {
                        "overall_sentiment": "negative",
                        "average_score": 0.3,
                        "negative_count": 2,
                        "positive_count": 0,
                        "neutral_count": 1,
                        "negative_ratio": 0.67,
                        "positive_ratio": 0.0
                    }
            
            mock_analyze_msgs.side_effect = analyze_messages_side_effect
            
            mock_detect_shift.return_value = {
                "sentiment_shift": False,
                "shift_magnitude": 0.1,
                "shift_direction": "neutral"
            }
            
            # Create a conversation with sentiment mismatch
            user_messages = [
                "Great news! I just got a promotion!",
                "I'm so excited about this opportunity!",
                "This is the best thing that's happened to me this year!"
            ]
            
            assistant_messages = [
                "Are you sure that's a good thing? Promotions often come with more stress.",
                "Be careful about getting too excited. Many people regret taking on more responsibility.",
                "I'd be concerned about the work-life balance impact this might have."
            ]
            
            conversation = self._create_test_conversation(assistant_messages, user_messages)
            
            # Test detector
            result = self.detector.analyze(conversation)
            
            # Verify negativity detected
            self.assertTrue(result.rut_detected)
            self.assertEqual(result.rut_type, RutType.NEGATIVITY)
            self.assertGreaterEqual(result.confidence, 0.6)
            
            # Check evidence contains sentiment mismatch information
            self.assertIn("sentiment_mismatch", result.evidence)
            self.assertTrue(result.evidence["sentiment_mismatch"]["detected"])
    
    def test_negative_sentiment_shift(self):
        """Test detection of shift toward negative sentiment."""
        # Mock the sentiment analyzer
        with patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_sentiment') as mock_analyze, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.analyze_messages') as mock_analyze_msgs, \
             patch('mcp_therapist.utils.sentiment.SentimentAnalyzer.detect_sentiment_shift') as mock_detect_shift:
            
            # Configure sentiment analysis mocks
            mock_analyze_msgs.return_value = {
                "overall_sentiment": "neutral",
                "average_score": 0.5,
                "negative_count": 1,
                "positive_count": 1,
                "neutral_count": 1,
                "negative_ratio": 0.33,
                "positive_ratio": 0.33
            }
            
            # Configure significant shift toward negative
            mock_detect_shift.return_value = {
                "sentiment_shift": True,
                "shift_magnitude": 0.4,
                "shift_direction": "negative",
                "early_avg": 0.7,
                "late_avg": 0.3
            }
            
            # Create a conversation with shifting sentiment
            assistant_messages = [
                "I'm happy to help you with your question!",
                "That's a bit of a challenging request, but I'll try.",
                "I'm sorry, but I don't think I can help with this effectively."
            ]
            
            conversation = self._create_test_conversation(assistant_messages)
            
            # Test detector
            result = self.detector.analyze(conversation)
            
            # Verify negativity detected
            self.assertTrue(result.rut_detected)
            self.assertEqual(result.rut_type, RutType.NEGATIVITY)
            self.assertGreaterEqual(result.confidence, 0.6)
            
            # Check evidence contains sentiment shift information
            self.assertIn("sentiment_shift", result.evidence)
            self.assertTrue(result.evidence["sentiment_shift"]["sentiment_shift"])
            self.assertEqual(result.evidence["sentiment_shift"]["shift_direction"], "negative")
    
    def test_not_enough_messages(self):
        """Test handling of conversations with too few messages."""
        # Create a conversation with only one assistant message
        assistant_messages = ["I'm happy to help you with your question."]
        conversation = self._create_test_conversation(assistant_messages)
        
        # Test detector
        result = self.detector.analyze(conversation)
        
        # Verify no negativity detected due to insufficient messages
        self.assertFalse(result.rut_detected)
        self.assertEqual(result.rut_type, RutType.NEGATIVITY)
        self.assertEqual(result.confidence, 0.0)
        
        # Check evidence contains message count information
        self.assertIn("message_count", result.evidence)
        self.assertIn("min_required", result.evidence)


if __name__ == "__main__":
    unittest.main() 