"""
MCP Resource providers for the MCP Therapist system.

This module contains provider classes that generate MCP resources
for various aspects of the therapy system.
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from mcp_therapist.mcp.client import MCPResource, MCPClient
from mcp_therapist.models.conversation import (
    Conversation, 
    Message,
    RutType,
    InterventionStrategy
)
from mcp_therapist.core.interventions.manager import InterventionManager
from mcp_therapist.utils.logging import logger


class ConversationResourceProvider:
    """
    Provider for conversation-related MCP resources.
    
    This provider generates resources from conversations, including
    conversation history, detected ruts, and intervention history.
    """
    
    def __init__(self, mcp_client: MCPClient):
        """
        Initialize the conversation resource provider.
        
        Args:
            mcp_client: The MCP client to register resources with
        """
        self.mcp_client = mcp_client
        self.logger = logger
    
    async def provide_conversation_resource(
        self,
        conversation: Conversation,
        window_size: Optional[int] = None
    ) -> str:
        """
        Provide a conversation as an MCP resource.
        
        Args:
            conversation: The conversation to provide
            window_size: Optional window size to limit the conversation history
            
        Returns:
            The resource ID
        """
        # If window size is specified, limit the conversation history
        if window_size and window_size > 0:
            messages = conversation.messages[-window_size:]
        else:
            messages = conversation.messages
        
        # Create the resource content
        content = {
            "id": conversation.id,
            "messages": [msg.to_dict() for msg in messages],
            "metadata": conversation.metadata,
            "created_at": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "message_count": len(messages),
            "total_message_count": len(conversation.messages),
            "window_size": window_size if window_size else len(conversation.messages),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"conversation_{conversation.id}_{uuid.uuid4()}",
            type="conversation_history",
            content=content,
            metadata=metadata,
            visibility="user_and_model"  # Both user and model can see the conversation
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided conversation resource: {resource_id}")
        
        return resource_id
    
    async def provide_rut_detection_resource(
        self,
        conversation: Conversation,
        detection_result: Dict[str, Any]
    ) -> str:
        """
        Provide a rut detection result as an MCP resource.
        
        Args:
            conversation: The conversation the detection is for
            detection_result: The detection result data
            
        Returns:
            The resource ID
        """
        # Create the resource content
        content = {
            "conversation_id": conversation.id,
            "detection_result": detection_result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "rut_type": detection_result.get("rut_type"),
            "confidence": detection_result.get("confidence"),
            "message_count": len(conversation.messages),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"rut_detection_{conversation.id}_{uuid.uuid4()}",
            type="rut_detection",
            content=content,
            metadata=metadata,
            visibility="model_only"  # Only the model needs to see this
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided rut detection resource: {resource_id}")
        
        return resource_id
    
    async def provide_intervention_history_resource(
        self,
        conversation: Conversation,
        intervention_manager: InterventionManager
    ) -> str:
        """
        Provide intervention history as an MCP resource.
        
        Args:
            conversation: The conversation to provide history for
            intervention_manager: The intervention manager to get history from
            
        Returns:
            The resource ID
        """
        # Get intervention history
        intervention_history = intervention_manager.get_intervention_history(conversation.id)
        
        # Create the resource content
        content = {
            "conversation_id": conversation.id,
            "interventions": intervention_history,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "intervention_count": len(intervention_history),
            "message_count": len(conversation.messages),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"intervention_history_{conversation.id}_{uuid.uuid4()}",
            type="intervention_history",
            content=content,
            metadata=metadata,
            visibility="model_only"  # Only the model needs to see this
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided intervention history resource: {resource_id}")
        
        return resource_id


class InterventionResourceProvider:
    """
    Provider for intervention-related MCP resources.
    
    This provider generates resources related to interventions,
    including strategies, evaluations, and statistics.
    """
    
    def __init__(self, mcp_client: MCPClient):
        """
        Initialize the intervention resource provider.
        
        Args:
            mcp_client: The MCP client to register resources with
        """
        self.mcp_client = mcp_client
        self.logger = logger
    
    async def provide_strategy_effectiveness_resource(
        self,
        intervention_manager: InterventionManager
    ) -> str:
        """
        Provide strategy effectiveness data as an MCP resource.
        
        Args:
            intervention_manager: The intervention manager to get data from
            
        Returns:
            The resource ID
        """
        # Get intervention statistics
        stats = intervention_manager.get_intervention_statistics()
        
        # Create the resource content
        content = {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "strategy_count": len(stats.get("by_strategy", {})),
            "rut_type_count": len(stats.get("by_rut_type", {})),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"strategy_effectiveness_{uuid.uuid4()}",
            type="strategy_effectiveness",
            content=content,
            metadata=metadata,
            visibility="model_only"  # Only the model needs to see this
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided strategy effectiveness resource: {resource_id}")
        
        return resource_id
    
    async def provide_intervention_evaluation_resource(
        self,
        conversation: Conversation,
        intervention_id: str,
        evaluation_result: Dict[str, Any]
    ) -> str:
        """
        Provide an intervention evaluation as an MCP resource.
        
        Args:
            conversation: The conversation the intervention is in
            intervention_id: The ID of the intervention
            evaluation_result: The evaluation result data
            
        Returns:
            The resource ID
        """
        # Create the resource content
        content = {
            "conversation_id": conversation.id,
            "intervention_id": intervention_id,
            "evaluation": evaluation_result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "success": evaluation_result.get("success"),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"intervention_evaluation_{intervention_id}_{uuid.uuid4()}",
            type="intervention_evaluation",
            content=content,
            metadata=metadata,
            visibility="model_only"  # Only the model needs to see this
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided intervention evaluation resource: {resource_id}")
        
        return resource_id
    
    async def provide_intervention_plan_resource(
        self,
        conversation: Conversation,
        intervention_plan: Dict[str, Any]
    ) -> str:
        """
        Provide an intervention plan as an MCP resource.
        
        Args:
            conversation: The conversation the plan is for
            intervention_plan: The intervention plan data
            
        Returns:
            The resource ID
        """
        # Create the resource content
        content = {
            "conversation_id": conversation.id,
            "plan": intervention_plan,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "strategy_type": intervention_plan.get("strategy_type"),
            "rut_type": intervention_plan.get("rut_type"),
            "confidence": intervention_plan.get("confidence"),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"intervention_plan_{conversation.id}_{uuid.uuid4()}",
            type="intervention_plan",
            content=content,
            metadata=metadata,
            visibility="model_only"  # Only the model needs to see this
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided intervention plan resource: {resource_id}")
        
        return resource_id


class UserProfileResourceProvider:
    """
    Provider for user profile-related MCP resources.
    
    This provider generates resources related to user profiles,
    including preferences, history, and personalization data.
    """
    
    def __init__(self, mcp_client: MCPClient):
        """
        Initialize the user profile resource provider.
        
        Args:
            mcp_client: The MCP client to register resources with
        """
        self.mcp_client = mcp_client
        self.logger = logger
    
    async def provide_user_preferences_resource(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> str:
        """
        Provide user preferences as an MCP resource.
        
        Args:
            user_id: The ID of the user
            preferences: The user preferences data
            
        Returns:
            The resource ID
        """
        # Create the resource content
        content = {
            "user_id": user_id,
            "preferences": preferences,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "preference_count": len(preferences),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"user_preferences_{user_id}_{uuid.uuid4()}",
            type="user_preferences",
            content=content,
            metadata=metadata,
            visibility="model_only"  # Only the model needs to see this
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided user preferences resource: {resource_id}")
        
        return resource_id
    
    async def provide_therapy_progress_resource(
        self,
        user_id: str,
        progress_data: Dict[str, Any]
    ) -> str:
        """
        Provide therapy progress data as an MCP resource.
        
        Args:
            user_id: The ID of the user
            progress_data: The therapy progress data
            
        Returns:
            The resource ID
        """
        # Create the resource content
        content = {
            "user_id": user_id,
            "progress": progress_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the resource metadata
        metadata = {
            "session_count": progress_data.get("session_count", 0),
            "last_session": progress_data.get("last_session", ""),
            "created_at": datetime.now().isoformat()
        }
        
        # Create and register the resource
        resource = MCPResource(
            id=f"therapy_progress_{user_id}_{uuid.uuid4()}",
            type="therapy_progress",
            content=content,
            metadata=metadata,
            visibility="model_only"  # Only the model needs to see this
        )
        
        resource_id = await self.mcp_client.register_resource(resource)
        self.logger.info(f"Provided therapy progress resource: {resource_id}")
        
        return resource_id 