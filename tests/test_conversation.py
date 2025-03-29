"""
Tests for the conversation models.
"""

import pytest
from datetime import datetime, timedelta

from mcp_therapist.models.conversation import (
    Conversation, 
    Message, 
    MessageRole, 
    RutType, 
    RutAnalysisResult,
    InterventionStrategy,
    InterventionPlan
)


def test_message_creation():
    """Test creating a message."""
    message = Message(role=MessageRole.USER, content="Hello!")
    
    assert message.role == MessageRole.USER
    assert message.content == "Hello!"
    assert isinstance(message.timestamp, datetime)
    assert isinstance(message.metadata, dict)
    assert len(message.metadata) == 0


def test_conversation_creation():
    """Test creating a conversation."""
    conversation = Conversation(id="test-conv-1")
    
    assert conversation.id == "test-conv-1"
    assert len(conversation.messages) == 0
    assert isinstance(conversation.metadata, dict)
    assert len(conversation.metadata) == 0
    assert isinstance(conversation.rut_analyses, list)
    assert len(conversation.rut_analyses) == 0
    assert isinstance(conversation.interventions, list)
    assert len(conversation.interventions) == 0


def test_add_message_to_conversation():
    """Test adding a message to a conversation."""
    conversation = Conversation(id="test-conv-1")
    
    # Add a user message
    message = conversation.add_message(MessageRole.USER, "Hello, AI!")
    
    assert len(conversation.messages) == 1
    assert conversation.messages[0] == message
    assert message.role == MessageRole.USER
    assert message.content == "Hello, AI!"
    
    # Add an assistant message
    message2 = conversation.add_message(MessageRole.ASSISTANT, "Hello, human!")
    
    assert len(conversation.messages) == 2
    assert conversation.messages[1] == message2
    assert message2.role == MessageRole.ASSISTANT
    assert message2.content == "Hello, human!"


def test_get_window():
    """Test getting a window of messages from a conversation."""
    conversation = Conversation(id="test-conv-1")
    
    # Add 10 messages
    for i in range(10):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        conversation.add_message(role, f"Message {i}")
    
    # Get a window of size 3
    window = conversation.get_window(size=3)
    
    assert len(window) == 3
    assert window[0].content == "Message 7"
    assert window[1].content == "Message 8"
    assert window[2].content == "Message 9"
    
    # Get a window larger than the conversation
    window = conversation.get_window(size=20)
    
    assert len(window) == 10
    assert window[0].content == "Message 0"
    assert window[9].content == "Message 9"


def test_get_messages_by_role():
    """Test getting messages by role."""
    conversation = Conversation(id="test-conv-1")
    
    # Add messages with different roles
    conversation.add_message(MessageRole.USER, "User message 1")
    conversation.add_message(MessageRole.ASSISTANT, "Assistant message 1")
    conversation.add_message(MessageRole.USER, "User message 2")
    conversation.add_message(MessageRole.THERAPIST, "Therapist message")
    conversation.add_message(MessageRole.ASSISTANT, "Assistant message 2")
    
    # Get user messages
    user_messages = conversation.get_messages_by_role(MessageRole.USER)
    
    assert len(user_messages) == 2
    assert user_messages[0].content == "User message 1"
    assert user_messages[1].content == "User message 2"
    
    # Get assistant messages
    assistant_messages = conversation.get_messages_by_role(MessageRole.ASSISTANT)
    
    assert len(assistant_messages) == 2
    assert assistant_messages[0].content == "Assistant message 1"
    assert assistant_messages[1].content == "Assistant message 2"
    
    # Get therapist messages
    therapist_messages = conversation.get_messages_by_role(MessageRole.THERAPIST)
    
    assert len(therapist_messages) == 1
    assert therapist_messages[0].content == "Therapist message"
    
    # Get system messages (there are none)
    system_messages = conversation.get_messages_by_role(MessageRole.SYSTEM)
    
    assert len(system_messages) == 0


def test_get_last_message():
    """Test getting the last message in a conversation."""
    conversation = Conversation(id="test-conv-1")
    
    # Empty conversation
    assert conversation.get_last_message() is None
    
    # Add a message
    conversation.add_message(MessageRole.USER, "First message")
    
    assert conversation.get_last_message().content == "First message"
    
    # Add another message
    conversation.add_message(MessageRole.ASSISTANT, "Second message")
    
    assert conversation.get_last_message().content == "Second message"


def test_rut_analysis_result():
    """Test creating a rut analysis result."""
    # No rut detected
    result = RutAnalysisResult()
    
    assert result.rut_detected is False
    assert result.rut_type is None
    assert result.confidence == 0.0
    assert len(result.evidence) == 0
    
    # Rut detected
    result = RutAnalysisResult(
        rut_detected=True,
        rut_type=RutType.REPETITION,
        confidence=0.85,
        evidence=["Similar response in message 3", "Similar response in message 5"]
    )
    
    assert result.rut_detected is True
    assert result.rut_type == RutType.REPETITION
    assert result.confidence == 0.85
    assert len(result.evidence) == 2
    assert "Similar response in message 3" in result.evidence


def test_intervention_plan():
    """Test creating an intervention plan."""
    plan = InterventionPlan(
        strategy=InterventionStrategy.REFRAME,
        target_topic="climate change",
        alternative_frame="opportunity for innovation"
    )
    
    assert plan.strategy == InterventionStrategy.REFRAME
    assert plan.target_topic == "climate change"
    assert plan.alternative_frame == "opportunity for innovation"
    assert isinstance(plan.metadata, dict)
    assert len(plan.metadata) == 0 