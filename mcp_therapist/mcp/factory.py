"""
Factory for creating MCP components.

This module provides factory functions for creating and initializing
MCP components with appropriate configuration.
"""

from typing import Optional, Dict, Any, Callable
import asyncio

from mcp_therapist.mcp.client import MCPClient
from mcp_therapist.mcp.resources import (
    ConversationResourceProvider,
    InterventionResourceProvider,
    UserProfileResourceProvider
)
from mcp_therapist.mcp.tools import InterventionTools, ConversationTools
from mcp_therapist.mcp.prompts import TherapyPromptManager
from mcp_therapist.mcp.sampling import MCPSamplingManager
from mcp_therapist.config import settings
from mcp_therapist.utils.logging import logger
from mcp_therapist.core.interventions.manager import InterventionManager


class MCPFactory:
    """
    Factory for creating and initializing MCP components.
    
    This class provides methods for creating MCP clients and
    related components with appropriate configuration.
    """
    
    @staticmethod
    async def create_client(
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> MCPClient:
        """
        Create and initialize an MCP client.
        
        Args:
            server_url: URL of the MCP server (defaults to settings)
            api_key: API key for the MCP server (defaults to settings)
            config: Additional configuration options
            
        Returns:
            Initialized MCP client
        """
        # Use defaults from settings if not provided
        server_url = server_url or settings.MCP_SERVER_URL
        api_key = api_key or settings.MCP_API_KEY
        
        # Create configuration dictionary
        client_config = {
            "connection_timeout": settings.MCP_CONNECTION_TIMEOUT,
            "request_timeout": settings.MCP_REQUEST_TIMEOUT,
            "reconnect_attempts": settings.MCP_RECONNECT_ATTEMPTS,
            "reconnect_backoff": settings.MCP_RECONNECT_BACKOFF,
            "keep_alive_interval": settings.MCP_KEEP_ALIVE_INTERVAL
        }
        
        # Update with any provided config
        if config:
            client_config.update(config)
        
        # Create the client
        client = MCPClient(server_url=server_url, api_key=api_key, config=client_config)
        
        # Initialize the connection
        try:
            await client.connect()
            logger.info(f"Connected to MCP server at {server_url}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
        
        return client
    
    @staticmethod
    async def create_resource_providers(
        client: MCPClient
    ) -> Dict[str, Any]:
        """
        Create all resource providers for the MCP client.
        
        Args:
            client: The MCP client to use
            
        Returns:
            Dictionary of resource providers
        """
        conversation_provider = ConversationResourceProvider(client)
        intervention_provider = InterventionResourceProvider(client)
        user_profile_provider = UserProfileResourceProvider(client)
        
        return {
            "conversation": conversation_provider,
            "intervention": intervention_provider,
            "user_profile": user_profile_provider
        }
    
    @staticmethod
    async def create_tool_providers(
        client: MCPClient,
        intervention_manager: Optional[InterventionManager] = None,
        get_conversation_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Create all tool providers for the MCP client.
        
        Args:
            client: The MCP client to use
            intervention_manager: Optional intervention manager
            get_conversation_func: Optional function to get conversations
            
        Returns:
            Dictionary of tool providers
        """
        tools = {}
        
        if intervention_manager:
            intervention_tools = InterventionTools(client, intervention_manager)
            tools["intervention"] = intervention_tools
        
        conversation_tools = ConversationTools(client, get_conversation_func)
        tools["conversation"] = conversation_tools
        
        return tools
    
    @staticmethod
    async def create_prompt_manager(
        client: MCPClient
    ) -> TherapyPromptManager:
        """
        Create a therapy prompt manager.
        
        Args:
            client: The MCP client to use
            
        Returns:
            Initialized therapy prompt manager
        """
        return TherapyPromptManager(client)
    
    @staticmethod
    async def create_sampling_manager(
        client: MCPClient
    ) -> MCPSamplingManager:
        """
        Create an MCP sampling manager.
        
        Args:
            client: The MCP client to use
            
        Returns:
            Initialized MCP sampling manager
        """
        return MCPSamplingManager(client)
    
    @staticmethod
    async def create_complete_mcp_system(
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        intervention_manager: Optional[InterventionManager] = None,
        get_conversation_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Create a complete MCP system with all components.
        
        This is a convenience method that creates all MCP components
        and returns them in a dictionary for easy access.
        
        Args:
            server_url: URL of the MCP server (defaults to settings)
            api_key: API key for the MCP server (defaults to settings)
            intervention_manager: Optional intervention manager
            get_conversation_func: Optional function to get conversations
            
        Returns:
            Dictionary of MCP components
        """
        # Check if MCP is enabled
        if not settings.MCP_ENABLED:
            logger.warning("MCP integration is disabled in settings.")
            return {
                "enabled": False,
                "client": None,
                "resource_providers": {},
                "tool_providers": {},
                "prompt_manager": None,
                "sampling_manager": None
            }
        
        # Create the client
        client = await MCPFactory.create_client(server_url, api_key)
        
        # Create all components
        resource_providers = await MCPFactory.create_resource_providers(client)
        tool_providers = await MCPFactory.create_tool_providers(
            client, intervention_manager, get_conversation_func
        )
        prompt_manager = await MCPFactory.create_prompt_manager(client)
        sampling_manager = await MCPFactory.create_sampling_manager(client)
        
        # Return everything in a dictionary
        return {
            "enabled": True,
            "client": client,
            "resource_providers": resource_providers,
            "tool_providers": tool_providers,
            "prompt_manager": prompt_manager,
            "sampling_manager": sampling_manager
        }
    
    @staticmethod
    async def shutdown_mcp_system(mcp_system: Dict[str, Any]) -> None:
        """
        Shut down an MCP system created with create_complete_mcp_system.
        
        Args:
            mcp_system: The MCP system to shut down
        """
        if not mcp_system.get("enabled", False):
            return
        
        client = mcp_system.get("client")
        if client:
            try:
                await client.disconnect()
                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.error(f"Error disconnecting from MCP server: {e}")


async def create_mcp_system(
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    intervention_manager: Optional[InterventionManager] = None,
    get_conversation_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Shorthand function to create a complete MCP system.
    
    Args:
        server_url: URL of the MCP server (defaults to settings)
        api_key: API key for the MCP server (defaults to settings)
        intervention_manager: Optional intervention manager
        get_conversation_func: Optional function to get conversations
        
    Returns:
        Dictionary of MCP components
    """
    return await MCPFactory.create_complete_mcp_system(
        server_url, api_key, intervention_manager, get_conversation_func
    )


async def shutdown_mcp_system(mcp_system: Dict[str, Any]) -> None:
    """
    Shorthand function to shut down an MCP system.
    
    Args:
        mcp_system: The MCP system to shut down
    """
    await MCPFactory.shutdown_mcp_system(mcp_system) 