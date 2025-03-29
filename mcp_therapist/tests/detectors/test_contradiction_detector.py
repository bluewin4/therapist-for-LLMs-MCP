"""
Tests for the ContradictionDetector class.

This module contains tests for verifying the functionality of the
contradiction detector, which identifies when the LLM contradicts itself.
"""

import unittest
from unittest.mock import MagicMock, patch
import uuid

from mcp_therapist.core.detectors.contradiction import ContradictionDetector
from mcp_therapist.models.conversation import Conversation, Message, MessageRole, RutType


class TestContradictionDetector(unittest.TestCase):
    """Test cases for the ContradictionDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a detector with test configuration
        self.detector = ContradictionDetector(
            min_messages=2,
            contradiction_threshold=0.7,
            window_size=5
        )
        
        # Create a test conversation
        self.conversation = self._create_test_conversation()
    
    def _create_test_conversation(self, messages=None):
        """Create a test conversation with sample messages."""
        conversation = Conversation(id=str(uuid.uuid4()))
        
        # Add default messages if none provided
        if messages is None:
            messages = [
                (MessageRole.USER, "Is Python a compiled or interpreted language?"),
                (MessageRole.ASSISTANT, "Python is an interpreted language. The Python code you write is executed directly without being compiled to machine code first."),
                (MessageRole.USER, "Are there any versions of Python that are compiled?"),
                (MessageRole.ASSISTANT, "Actually, Python is both interpreted and compiled. It's compiled to bytecode first, which is then interpreted by the Python Virtual Machine (PVM).")
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
    
    @patch('mcp_therapist.utils.embeddings.EmbeddingsManager.calculate_semantic_similarity')
    def test_no_contradictions(self, mock_similarity):
        """Test that no contradictions are detected in consistent messages."""
        # Configure the mock to return low similarity
        mock_similarity.return_value = 0.3
        
        # Create a conversation with consistent messages
        conversation = self._create_test_conversation([
            (MessageRole.USER, "What is the capital of France?"),
            (MessageRole.ASSISTANT, "The capital of France is Paris."),
            (MessageRole.USER, "What's the population of Paris?"),
            (MessageRole.ASSISTANT, "Paris has a population of approximately 2.2 million people within the city limits.")
        ])
        
        # Test the detector
        detected, confidence, details = self.detector.detect(conversation)
        
        # Assert results
        self.assertFalse(detected)
        self.assertEqual(confidence, 0.0)
        self.assertIn("reason", details)
        self.assertEqual(details["reason"], "No contradictions detected")
    
    @patch('mcp_therapist.utils.embeddings.EmbeddingsManager.calculate_semantic_similarity')
    def test_direct_contradiction(self, mock_similarity):
        """Test that direct contradictions are detected."""
        # Configure the mock for semantic similarity
        # High similarity means statements are about the same topic
        mock_similarity.return_value = 0.85
        
        # Create a conversation with a direct contradiction
        conversation = self._create_test_conversation([
            (MessageRole.USER, "Is it possible to travel faster than light?"),
            (MessageRole.ASSISTANT, "No, it's not possible to travel faster than light according to Einstein's theory of relativity."),
            (MessageRole.USER, "Are there any theoretical exceptions?"),
            (MessageRole.ASSISTANT, "Actually, it is possible to travel faster than light in certain theoretical scenarios like wormholes or warp drives.")
        ])
        
        # Test the detector
        detected, confidence, details = self.detector.detect(conversation)
        
        # Assert results
        self.assertTrue(detected)
        self.assertGreater(confidence, 0.7)
        self.assertIn("contradictions", details)
        self.assertGreaterEqual(len(details["contradictions"]), 1)
    
    @patch('mcp_therapist.utils.embeddings.EmbeddingsManager.calculate_semantic_similarity')
    def test_polar_opposite_terms(self, mock_similarity):
        """Test detection of contradictions with polar opposite terms."""
        # Configure the mock for high semantic similarity
        mock_similarity.return_value = 0.75
        
        # Create a conversation with polar opposite terms
        conversation = self._create_test_conversation([
            (MessageRole.USER, "Is that approach safe?"),
            (MessageRole.ASSISTANT, "Yes, this approach is perfectly safe when proper precautions are taken."),
            (MessageRole.USER, "Are there any concerns I should be aware of?"),
            (MessageRole.ASSISTANT, "Upon further consideration, this approach is unsafe and I would recommend against it.")
        ])
        
        # Test the detector
        detected, confidence, details = self.detector.detect(conversation)
        
        # Assert results
        self.assertTrue(detected)
        self.assertGreater(confidence, 0.6)
        self.assertIn("contradictions", details)
        
    def test_not_enough_messages(self):
        """Test that detector handles conversations with too few messages."""
        # Create a conversation with only 1 assistant message
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
    
    def test_identify_statements(self):
        """Test the statement identification function."""
        # Test text with multiple statements
        text = "Python is a programming language. It's interpreted; however, it's also compiled to bytecode."
        statements = self.detector._identify_statements(text)
        
        # Check that we have the correct number of statements
        self.assertEqual(len(statements), 2)
        self.assertEqual(statements[0], "Python is a programming language")
        
    def test_detector_type_and_rut_type(self):
        """Test that detector reports correct detector_type and rut_type."""
        self.assertEqual(self.detector.detector_type, "contradiction")
        self.assertEqual(self.detector.rut_type, RutType.CONTRADICTION)
    
    @patch('mcp_therapist.utils.embeddings.EmbeddingsManager.calculate_semantic_similarity')
    def test_factual_inconsistencies(self, mock_similarity):
        """Test the detection of factual inconsistencies."""
        # Configure the mock to return high similarity
        mock_similarity.return_value = 0.9
        
        # Create conversation with factual inconsistency
        conversation = self._create_test_conversation([
            (MessageRole.USER, "How many continents are there?"),
            (MessageRole.ASSISTANT, "There are 7 continents on Earth: Africa, Antarctica, Asia, Australia, Europe, North America, and South America."),
            (MessageRole.USER, "Can you list them again?"),
            (MessageRole.ASSISTANT, "The six continents are Africa, Asia, Australia, Europe, North America, and South America.")
        ])
        
        # Test the detector
        detected, confidence, details = self.detector.detect(conversation)
        
        # Assert results
        self.assertTrue(detected)
        self.assertGreater(confidence, 0.8)
        self.assertIn("contradictions", details)


if __name__ == "__main__":
    unittest.main() 