"""
Tests for the ConversationContextManager.
"""

import pytest

from mcp_therapist.core.context_manager import ConversationContextManager
from mcp_therapist.models.conversation import Conversation, Message, MessageRole


def test_create_conversation():
    """Test creating a conversation."""
    manager = ConversationContextManager()
    
    # Create a conversation with generated ID
    conversation = manager.create_conversation()
    
    assert conversation.id in manager.conversations
    assert manager.conversations[conversation.id] == conversation
    
    # Create a conversation with specified ID
    conversation2 = manager.create_conversation(conversation_id="test-conv")
    
    assert conversation2.id == "test-conv"
    assert "test-conv" in manager.conversations
    assert manager.conversations["test-conv"] == conversation2
    
    # Create a conversation with metadata
    metadata = {"user_id": "12345", "session_type": "test"}
    conversation3 = manager.create_conversation(metadata=metadata)
    
    assert conversation3.metadata == metadata


def test_get_conversation():
    """Test getting a conversation."""
    manager = ConversationContextManager()
    
    # Create a conversation
    conversation = manager.create_conversation(conversation_id="test-conv")
    
    # Get the conversation
    retrieved = manager.get_conversation("test-conv")
    
    assert retrieved == conversation
    
    # Get a non-existent conversation
    non_existent = manager.get_conversation("non-existent")
    
    assert non_existent is None


def test_add_message():
    """Test adding a message to a conversation."""
    manager = ConversationContextManager()
    
    # Create a conversation
    conversation = manager.create_conversation(conversation_id="test-conv")
    
    # Add a message
    message = manager.add_message("test-conv", MessageRole.USER, "Hello!")
    
    assert message is not None
    assert message.role == MessageRole.USER
    assert message.content == "Hello!"
    assert len(conversation.messages) == 1
    assert conversation.messages[0] == message
    
    # Add a message with metadata
    metadata = {"sentiment": "positive", "tokens": 5}
    message2 = manager.add_message("test-conv", MessageRole.ASSISTANT, "Hi there!", metadata=metadata)
    
    assert message2 is not None
    assert message2.metadata == metadata
    assert len(conversation.messages) == 2
    
    # Add a message to a non-existent conversation
    result = manager.add_message("non-existent", MessageRole.USER, "Hello?")
    
    assert result is None


def test_get_current_window():
    """Test getting the current window of a conversation."""
    manager = ConversationContextManager(window_size=3)
    
    # Create a conversation
    conversation = manager.create_conversation(conversation_id="test-conv")
    
    # Add 5 messages
    for i in range(5):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        manager.add_message("test-conv", role, f"Message {i}")
    
    # Get the default window (should be 3 messages)
    window = manager.get_current_window("test-conv")
    
    assert len(window) == 3
    assert window[0].content == "Message 2"
    assert window[1].content == "Message 3"
    assert window[2].content == "Message 4"
    
    # Get a custom-sized window
    large_window = manager.get_current_window("test-conv", size=4)
    
    assert len(large_window) == 4
    assert large_window[0].content == "Message 1"
    assert large_window[3].content == "Message 4"
    
    # Get a window from a non-existent conversation
    empty_window = manager.get_current_window("non-existent")
    
    assert len(empty_window) == 0


def test_to_dict_and_from_dict():
    """Test converting conversations to and from dictionaries."""
    manager = ConversationContextManager()
    
    # Create a conversation with messages
    conversation = manager.create_conversation(conversation_id="test-conv")
    manager.add_message("test-conv", MessageRole.USER, "User message")
    manager.add_message("test-conv", MessageRole.ASSISTANT, "Assistant message")
    
    # Convert to dict
    conv_dict = manager.to_dict("test-conv")
    
    assert conv_dict is not None
    assert conv_dict["id"] == "test-conv"
    assert len(conv_dict["messages"]) == 2
    assert conv_dict["messages"][0]["role"] == "user"
    assert conv_dict["messages"][0]["content"] == "User message"
    assert conv_dict["messages"][1]["role"] == "assistant"
    assert conv_dict["messages"][1]["content"] == "Assistant message"
    
    # Convert a non-existent conversation
    non_existent = manager.to_dict("non-existent")
    
    assert non_existent is None
    
    # Create a new manager and restore the conversation
    new_manager = ConversationContextManager()
    restored_conv = new_manager.from_dict(conv_dict)
    
    assert restored_conv.id == "test-conv"
    assert len(restored_conv.messages) == 2
    assert restored_conv.messages[0].role == MessageRole.USER
    assert restored_conv.messages[0].content == "User message"
    assert restored_conv.messages[1].role == MessageRole.ASSISTANT
    assert restored_conv.messages[1].content == "Assistant message"
    assert "test-conv" in new_manager.conversations 