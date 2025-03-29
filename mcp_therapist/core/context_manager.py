import uuid
from typing import Dict, List, Optional

from mcp_therapist.models.conversation import Conversation, Message, MessageRole


class ConversationContextManager:
    """Manages conversation contexts and provides access to conversation data."""
    
    def __init__(self, window_size: int = 5):
        """Initialize the context manager.
        
        Args:
            window_size: Default number of messages to include in context window.
        """
        self.conversations: Dict[str, Conversation] = {}
        self.window_size = window_size
    
    def create_conversation(self, conversation_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Conversation:
        """Create a new conversation.
        
        Args:
            conversation_id: Optional ID for the conversation. If not provided, a UUID will be generated.
            metadata: Optional metadata to associate with the conversation.
            
        Returns:
            The newly created Conversation object.
        """
        conversation_id = conversation_id or str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            metadata=metadata or {}
        )
        self.conversations[conversation_id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation to retrieve.
            
        Returns:
            The Conversation object if found, None otherwise.
        """
        return self.conversations.get(conversation_id)
    
    def add_message(self, conversation_id: str, role: MessageRole, content: str, 
                   metadata: Optional[Dict] = None) -> Optional[Message]:
        """Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation to add the message to.
            role: Role of the message sender.
            content: Content of the message.
            metadata: Optional metadata to associate with the message.
            
        Returns:
            The added Message object if successful, None if the conversation was not found.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        return conversation.add_message(role, content, metadata)
    
    def get_current_window(self, conversation_id: str, size: Optional[int] = None) -> List[Message]:
        """Get the current context window for a conversation.
        
        Args:
            conversation_id: ID of the conversation to get the window for.
            size: Optional size of the window. If not provided, the default window size will be used.
            
        Returns:
            A list of the most recent messages up to the specified size, or an empty list if
            the conversation was not found.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        return conversation.get_window(size or self.window_size)
    
    def to_dict(self, conversation_id: str) -> Optional[Dict]:
        """Convert a conversation to a dictionary for serialization.
        
        Args:
            conversation_id: ID of the conversation to convert.
            
        Returns:
            Dictionary representation of the conversation if found, None otherwise.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        return conversation.model_dump()
    
    def from_dict(self, data: Dict) -> Conversation:
        """Create a conversation from a dictionary.
        
        Args:
            data: Dictionary representation of a conversation.
            
        Returns:
            The created Conversation object.
        """
        conversation = Conversation.model_validate(data)
        self.conversations[conversation.id] = conversation
        return conversation 