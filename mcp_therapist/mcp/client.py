"""
MCP Client for the MCP Therapist system.

This module provides a comprehensive Model Context Protocol (MCP) client
implementation that supports resources, tools, prompts, and sampling features.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, Generic

from mcp_therapist.config.settings import settings
from mcp_therapist.models.conversation import Conversation, Message
from mcp_therapist.utils.logging import logger


T = TypeVar('T')


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Exception raised when connection to MCP server fails."""
    pass


class MCPCapabilityError(MCPError):
    """Exception raised when requested capability is not supported."""
    pass


class MCPPermissionError(MCPError):
    """Exception raised when permission is denied for a requested operation."""
    pass


class MCPResource(Generic[T]):
    """
    Represents an MCP resource that can be shared with language models.
    
    Resources are contextual information that can be presented to the user
    or sent directly to the language model.
    """
    
    def __init__(
        self,
        id: str,
        type: str,
        content: T,
        metadata: Optional[Dict[str, Any]] = None,
        visibility: str = "user_and_model"
    ):
        """
        Initialize an MCP resource.
        
        Args:
            id: Unique identifier for the resource
            type: Resource type (e.g., "text", "conversation", "intervention_stats")
            content: The actual content of the resource
            metadata: Additional metadata about the resource
            visibility: Who can see the resource ("user_and_model", "model_only", "user_only")
        """
        self.id = id
        self.type = type
        self.content = content
        self.metadata = metadata or {}
        self.visibility = visibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the resource to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "visibility": self.visibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResource':
        """Create a resource from a dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            visibility=data.get("visibility", "user_and_model")
        )


class MCPTool:
    """
    Represents an MCP tool that can be exposed to language models.
    
    Tools allow the language model to perform actions in the external world,
    such as retrieving information, manipulating data, or interacting with
    other systems.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        requires_confirmation: bool = True
    ):
        """
        Initialize an MCP tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            parameters: JSON Schema for the tool's parameters
            handler: Function to call when the tool is invoked
            requires_confirmation: Whether user confirmation is required before execution
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.requires_confirmation = requires_confirmation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation
        }
    
    async def invoke(self, params: Dict[str, Any], user_confirmed: bool = False) -> Any:
        """
        Invoke the tool with the given parameters.
        
        Args:
            params: Parameters for the tool invocation
            user_confirmed: Whether the user has confirmed this invocation
            
        Returns:
            The result of the tool invocation
            
        Raises:
            MCPPermissionError: If user confirmation is required but not provided
        """
        if self.requires_confirmation and not user_confirmed:
            raise MCPPermissionError(
                f"Tool {self.name} requires user confirmation before execution."
            )
        
        return await self.handler(**params)


class MCPPrompt:
    """
    Represents an MCP prompt template for standardized interactions.
    
    Prompts are templated messages that can be presented to the user
    or used to structure interactions with the language model.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        template: str,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize an MCP prompt template.
        
        Args:
            id: Unique identifier for the prompt
            name: Human-readable name for the prompt
            template: The prompt template with parameter placeholders
            parameters: Parameter schema for the prompt
            metadata: Additional metadata for the prompt
        """
        self.id = id
        self.name = name
        self.template = template
        self.parameters = parameters or {}
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the prompt to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "parameters": self.parameters,
            "metadata": self.metadata
        }
    
    def render(self, params: Dict[str, Any]) -> str:
        """
        Render the prompt template with the given parameters.
        
        Args:
            params: Parameters to fill into the template
            
        Returns:
            The rendered prompt string
        """
        # Simple string formatting for templates
        # In a real implementation, this would use a more robust template engine
        rendered = self.template
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            rendered = rendered.replace(placeholder, str(value))
        
        return rendered


class MCPClient:
    """
    Client for interacting with MCP servers.
    
    This client supports all MCP features including resources, tools,
    prompts, and sampling.
    """
    
    def __init__(self, server_url: Optional[str] = None):
        """
        Initialize the MCP client.
        
        Args:
            server_url: URL of the MCP server to connect to
        """
        self.server_url = server_url or settings.MCP_SERVER_URL
        self.logger = logger
        self.session_id = str(uuid.uuid4())
        self.capabilities = {}
        self.resources: Dict[str, MCPResource] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.connected = False
    
    async def connect(self) -> bool:
        """
        Connect to the MCP server and negotiate capabilities.
        
        Returns:
            True if connection was successful, False otherwise
        """
        if not self.server_url:
            self.logger.warning("No MCP server URL configured. MCP features will be disabled.")
            return False
        
        try:
            # In a real implementation, this would make an HTTP request to the server
            # For now, we'll simulate a successful connection
            self.connected = True
            self.capabilities = {
                "resources": {
                    "version": "1.0",
                    "supports_visibility_control": True
                },
                "tools": {
                    "version": "1.0",
                    "supports_user_confirmation": True
                },
                "prompts": {
                    "version": "1.0"
                },
                "sampling": {
                    "version": "1.0",
                    "supports_server_filtering": True
                }
            }
            
            self.logger.info(f"Connected to MCP server: {self.server_url}")
            self.logger.debug(f"Negotiated capabilities: {self.capabilities}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.connected:
            # In a real implementation, this would clean up the connection
            self.connected = False
            self.logger.info("Disconnected from MCP server")
    
    async def register_resource(self, resource: MCPResource) -> str:
        """
        Register a resource with the MCP server.
        
        Args:
            resource: The resource to register
            
        Returns:
            The resource ID
            
        Raises:
            MCPConnectionError: If not connected to the server
            MCPCapabilityError: If resources are not supported
        """
        self._check_connection()
        self._check_capability("resources")
        
        # In a real implementation, this would send the resource to the server
        # For now, we'll just store it locally
        self.resources[resource.id] = resource
        
        self.logger.info(f"Registered resource: {resource.id} ({resource.type})")
        return resource.id
    
    async def get_resource(self, resource_id: str) -> MCPResource:
        """
        Get a resource by ID.
        
        Args:
            resource_id: The ID of the resource to get
            
        Returns:
            The requested resource
            
        Raises:
            MCPConnectionError: If not connected to the server
            KeyError: If the resource does not exist
        """
        self._check_connection()
        
        if resource_id not in self.resources:
            raise KeyError(f"Resource not found: {resource_id}")
        
        return self.resources[resource_id]
    
    async def update_resource(self, resource: MCPResource) -> None:
        """
        Update an existing resource.
        
        Args:
            resource: The resource to update
            
        Raises:
            MCPConnectionError: If not connected to the server
            KeyError: If the resource does not exist
        """
        self._check_connection()
        
        if resource.id not in self.resources:
            raise KeyError(f"Resource not found: {resource.id}")
        
        # In a real implementation, this would update the resource on the server
        self.resources[resource.id] = resource
        
        self.logger.info(f"Updated resource: {resource.id}")
    
    async def delete_resource(self, resource_id: str) -> None:
        """
        Delete a resource by ID.
        
        Args:
            resource_id: The ID of the resource to delete
            
        Raises:
            MCPConnectionError: If not connected to the server
            KeyError: If the resource does not exist
        """
        self._check_connection()
        
        if resource_id not in self.resources:
            raise KeyError(f"Resource not found: {resource_id}")
        
        # In a real implementation, this would delete the resource from the server
        del self.resources[resource_id]
        
        self.logger.info(f"Deleted resource: {resource_id}")
    
    async def register_tool(self, tool: MCPTool) -> None:
        """
        Register a tool with the MCP server.
        
        Args:
            tool: The tool to register
            
        Raises:
            MCPConnectionError: If not connected to the server
            MCPCapabilityError: If tools are not supported
        """
        self._check_connection()
        self._check_capability("tools")
        
        # In a real implementation, this would register the tool with the server
        self.tools[tool.name] = tool
        
        self.logger.info(f"Registered tool: {tool.name}")
    
    async def invoke_tool(
        self, 
        tool_name: str, 
        params: Dict[str, Any],
        user_confirmed: bool = False
    ) -> Any:
        """
        Invoke a tool with the given parameters.
        
        Args:
            tool_name: The name of the tool to invoke
            params: Parameters for the tool invocation
            user_confirmed: Whether the user has confirmed this invocation
            
        Returns:
            The result of the tool invocation
            
        Raises:
            MCPConnectionError: If not connected to the server
            KeyError: If the tool does not exist
            MCPPermissionError: If user confirmation is required but not provided
        """
        self._check_connection()
        
        if tool_name not in self.tools:
            raise KeyError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        
        self.logger.info(f"Invoking tool: {tool_name}")
        self.logger.debug(f"Tool parameters: {params}")
        
        return await tool.invoke(params, user_confirmed)
    
    async def register_prompt(self, prompt: MCPPrompt) -> str:
        """
        Register a prompt template with the MCP server.
        
        Args:
            prompt: The prompt template to register
            
        Returns:
            The prompt ID
            
        Raises:
            MCPConnectionError: If not connected to the server
            MCPCapabilityError: If prompts are not supported
        """
        self._check_connection()
        self._check_capability("prompts")
        
        # In a real implementation, this would register the prompt with the server
        self.prompts[prompt.id] = prompt
        
        self.logger.info(f"Registered prompt: {prompt.id} ({prompt.name})")
        return prompt.id
    
    async def get_prompt(self, prompt_id: str) -> MCPPrompt:
        """
        Get a prompt template by ID.
        
        Args:
            prompt_id: The ID of the prompt to get
            
        Returns:
            The requested prompt template
            
        Raises:
            MCPConnectionError: If not connected to the server
            KeyError: If the prompt does not exist
        """
        self._check_connection()
        
        if prompt_id not in self.prompts:
            raise KeyError(f"Prompt not found: {prompt_id}")
        
        return self.prompts[prompt_id]
    
    async def render_prompt(self, prompt_id: str, params: Dict[str, Any]) -> str:
        """
        Render a prompt template with the given parameters.
        
        Args:
            prompt_id: The ID of the prompt to render
            params: Parameters for the prompt template
            
        Returns:
            The rendered prompt string
            
        Raises:
            MCPConnectionError: If not connected to the server
            KeyError: If the prompt does not exist
        """
        prompt = await self.get_prompt(prompt_id)
        
        self.logger.debug(f"Rendering prompt: {prompt_id}")
        rendered = prompt.render(params)
        
        return rendered
    
    async def sample_llm(
        self,
        prompt: str,
        params: Dict[str, Any] = None,
        visibility: str = "server_only"
    ) -> str:
        """
        Perform server-side LLM sampling with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            params: Additional parameters for the sampling request
            visibility: Who can see the prompt ("server_only", "user_and_server")
            
        Returns:
            The generated response from the LLM
            
        Raises:
            MCPConnectionError: If not connected to the server
            MCPCapabilityError: If sampling is not supported
            MCPPermissionError: If the user has not approved sampling
        """
        self._check_connection()
        self._check_capability("sampling")
        
        if visibility not in ["server_only", "user_and_server"]:
            raise ValueError(f"Invalid visibility: {visibility}")
        
        # In a real implementation, this would send the sampling request to the server
        # For now, we'll simulate a response
        
        self.logger.info("Performing server-side LLM sampling")
        self.logger.debug(f"Prompt visibility: {visibility}")
        
        # Simulate a response - in a real implementation, this would come from the server
        response = "This is a simulated response from the LLM"
        
        return response
    
    async def create_conversation_resource(self, conversation: Conversation) -> str:
        """
        Create a resource from a conversation.
        
        This is a convenience method for creating a resource from a conversation object.
        
        Args:
            conversation: The conversation to create a resource from
            
        Returns:
            The resource ID
        """
        resource = MCPResource(
            id=f"conversation_{conversation.id}",
            type="conversation",
            content={
                "id": conversation.id,
                "messages": [msg.to_dict() for msg in conversation.messages],
                "metadata": conversation.metadata
            },
            metadata={
                "message_count": len(conversation.messages),
                "created_at": conversation.metadata.get("created_at", "")
            },
            visibility="user_and_model"
        )
        
        return await self.register_resource(resource)
    
    async def create_intervention_stats_resource(self, stats: Dict[str, Any]) -> str:
        """
        Create a resource from intervention statistics.
        
        This is a convenience method for creating a resource from intervention statistics.
        
        Args:
            stats: The intervention statistics to create a resource from
            
        Returns:
            The resource ID
        """
        resource = MCPResource(
            id=f"intervention_stats_{uuid.uuid4()}",
            type="intervention_stats",
            content=stats,
            metadata={
                "created_at": stats.get("timestamp", "")
            },
            visibility="model_only"  # Users don't need to see the raw stats
        )
        
        return await self.register_resource(resource)
    
    def _check_connection(self) -> None:
        """Check if connected to the server and raise an exception if not."""
        if not self.connected:
            raise MCPConnectionError("Not connected to MCP server")
    
    def _check_capability(self, capability: str) -> None:
        """
        Check if the server supports the given capability.
        
        Args:
            capability: The capability to check
            
        Raises:
            MCPCapabilityError: If the capability is not supported
        """
        if capability not in self.capabilities:
            raise MCPCapabilityError(f"MCP server does not support: {capability}") 