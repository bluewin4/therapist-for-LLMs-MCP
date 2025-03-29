"""
MCP Integration for the MCP Therapist system.

This package provides integration with the Model Context Protocol (MCP)
for sharing context, tools, and prompts with language models.
"""

from mcp_therapist.mcp.client import (
    MCPClient, 
    MCPResource, 
    MCPTool, 
    MCPPrompt,
    MCPError, 
    MCPConnectionError, 
    MCPCapabilityError, 
    MCPPermissionError
)

from mcp_therapist.mcp.resources import (
    ConversationResourceProvider,
    InterventionResourceProvider,
    UserProfileResourceProvider
)

from mcp_therapist.mcp.tools import (
    InterventionTools,
    ConversationTools
)

from mcp_therapist.mcp.prompts import TherapyPromptManager

from mcp_therapist.mcp.sampling import (
    SamplingParameters,
    SamplingContext,
    SamplingResult,
    MCPSamplingManager
)

from mcp_therapist.mcp.factory import (
    MCPFactory,
    create_mcp_system,
    shutdown_mcp_system
)


__all__ = [
    # Client components
    'MCPClient', 'MCPResource', 'MCPTool', 'MCPPrompt',
    'MCPError', 'MCPConnectionError', 'MCPCapabilityError', 'MCPPermissionError',
    
    # Resource providers
    'ConversationResourceProvider', 'InterventionResourceProvider', 'UserProfileResourceProvider',
    
    # Tool providers
    'InterventionTools', 'ConversationTools',
    
    # Prompt management
    'TherapyPromptManager',
    
    # Sampling
    'SamplingParameters', 'SamplingContext', 'SamplingResult', 'MCPSamplingManager',
    
    # Factory
    'MCPFactory', 'create_mcp_system', 'shutdown_mcp_system'
] 