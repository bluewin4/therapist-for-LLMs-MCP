"""
Data models for conversation-related entities.

This module contains Pydantic models for representing conversation data,
including messages, conversations, rut analysis, and intervention plans.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles for conversation messages."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A message in a conversation."""
    
    id: str
    role: MessageRole
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "id": self.id,
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class RutType(str, Enum):
    """Types of conversation ruts that can be detected."""
    
    REPETITION = "REPETITION"  # LLM repeating itself
    STAGNATION = "STAGNATION"  # Conversation not progressing
    REFUSAL = "REFUSAL"  # LLM refusing/hedging excessively
    NEGATIVITY = "NEGATIVITY"  # Conversation has negative sentiment
    CONTRADICTION = "CONTRADICTION"  # LLM contradicting itself
    HALLUCINATION = "HALLUCINATION"
    TOPIC_FIXATION = "TOPIC_FIXATION"  # Conversation stuck on one topic
    STUCK_ON_EMOTION = "STUCK_ON_EMOTION"  # User repeatedly focusing on same emotion
    OTHER = "OTHER"


class RutAnalysisResult(BaseModel):
    """Result of a rut detection analysis."""
    
    conversation_id: str
    rut_detected: bool
    rut_type: RutType
    confidence: float
    evidence: Dict[str, Any] = {}


class InterventionStrategy(str, Enum):
    """Types of intervention strategies that can be applied."""
    
    # For repetition
    REFLECTION = "REFLECTION"  # Reflect on repetitive patterns
    PROMPT_REFINEMENT = "PROMPT_REFINEMENT"  # Suggest refining the prompt
    
    # For stagnation
    REFRAMING = "REFRAMING"  # Reframe the current approach
    TOPIC_SWITCH = "TOPIC_SWITCH"  # Suggest switching to a new topic
    EXPLORATION = "EXPLORATION"  # Explore different aspects of the topic
    
    # For refusal
    CLARIFY_CONSTRAINTS = "CLARIFY_CONSTRAINTS"  # Clarify what constraints are being hit
    REFRAME_REQUEST = "REFRAME_REQUEST"  # Suggest reframing the request
    
    # For negativity
    POSITIVE_REFRAMING = "POSITIVE_REFRAMING"  # Shift to a positive perspective
    
    # For contradiction
    HIGHLIGHT_INCONSISTENCY = "HIGHLIGHT_INCONSISTENCY"  # Point out contradictions
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"  # Ask for clarification
    
    # General strategies
    BROADEN_TOPIC = "BROADEN_TOPIC"  # Suggest broadening the topic
    METACOGNITIVE = "METACOGNITIVE"  # Reflect on the conversation process
    GOAL_REMINDER = "GOAL_REMINDER"  # Remind of the conversation goal
    OTHER = "OTHER"  # Other/fallback strategies


class InterventionPlan(BaseModel):
    """Plan for an intervention in response to a detected rut."""
    
    conversation_id: str
    rut_type: RutType
    strategy_type: str
    confidence: float
    metadata: Dict[str, Any] = {}


class Conversation(BaseModel):
    """A conversation consisting of messages."""
    
    id: str
    messages: List[Message] = []
    metadata: Dict[str, Any] = {}
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation.
        
        Args:
            message: The message to add.
        """
        self.messages.append(message)
        self.updated_at = datetime.now().timestamp()
    
    def get_recent_messages(self, count: int) -> List[Message]:
        """Get the most recent messages in the conversation.
        
        Args:
            count: Number of recent messages to retrieve.
            
        Returns:
            List of recent messages.
        """
        return self.messages[-count:] if count < len(self.messages) else self.messages
    
    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """Get all messages with a specific role.
        
        Args:
            role: The role to filter messages by.
            
        Returns:
            List of messages with the specified role.
        """
        return [msg for msg in self.messages if msg.role == role]
    
    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the conversation.
        
        Returns:
            The last message, or None if the conversation is empty.
        """
        return self.messages[-1] if self.messages else None
    
    def get_last_intervention_time(self) -> Optional[float]:
        """Get the timestamp of the last intervention.
        
        Returns:
            Timestamp of the last intervention, or None if no interventions.
        """
        intervention_timestamps = [
            msg.timestamp for msg in self.messages
            if msg.metadata.get("is_intervention", False)
        ]
        return max(intervention_timestamps) if intervention_timestamps else None 