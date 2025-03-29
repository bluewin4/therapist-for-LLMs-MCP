from typing import Any, Dict, List, Optional, Union
import json
import uuid

import jsonrpcclient
import requests
from pydantic import BaseModel


class MCPCapability(BaseModel):
    """Represents a capability provided by an MCP server."""
    name: str
    version: str
    optional: bool = False


class MCPClient:
    """Basic Model Context Protocol (MCP) client implementation.
    
    Handles JSON-RPC 2.0 communication with MCP servers and provides
    methods for MCP-compatible operations.
    """
    
    def __init__(self, server_url: str):
        """Initialize the MCP client.
        
        Args:
            server_url: URL of the MCP server.
        """
        self.server_url = server_url
        self.session = requests.Session()
        self.capabilities: List[MCPCapability] = []
        self.server_capabilities: List[MCPCapability] = []
        
        # Initialize with default client capabilities
        self._init_capabilities()
    
    def _init_capabilities(self):
        """Initialize the default client capabilities."""
        self.capabilities = [
            MCPCapability(name="resources", version="1.0"),
            MCPCapability(name="tools", version="1.0"),
            MCPCapability(name="prompts", version="1.0"),
        ]
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID for JSON-RPC requests.
        
        Returns:
            A unique request ID.
        """
        return str(uuid.uuid4())
    
    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server.
        
        Args:
            method: The method to call on the server.
            params: Optional parameters to include in the request.
            
        Returns:
            The response from the server.
            
        Raises:
            ValueError: If the request fails or returns an error.
        """
        # Create the JSON-RPC request
        request = jsonrpcclient.request(method, params or {})
        
        # Send the request to the server
        response = self.session.post(
            self.server_url,
            json=request,
            headers={"Content-Type": "application/json"}
        )
        
        # Parse the response
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")
        
        result = response.json()
        
        # Check for errors
        if "error" in result:
            error = result["error"]
            raise ValueError(f"Server returned error: {error.get('message', 'Unknown error')}")
        
        return result.get("result", {})
    
    def negotiate_capabilities(self) -> List[MCPCapability]:
        """Negotiate capabilities with the MCP server.
        
        Returns:
            List of capabilities supported by both the client and server.
            
        Raises:
            ValueError: If capability negotiation fails.
        """
        try:
            # Get server capabilities
            result = self._send_request("initialize", {
                "capabilities": [cap.model_dump() for cap in self.capabilities]
            })
            
            # Parse server capabilities
            server_caps = result.get("capabilities", [])
            self.server_capabilities = [MCPCapability(**cap) for cap in server_caps]
            
            # Find common capabilities
            common_capabilities = []
            for client_cap in self.capabilities:
                for server_cap in self.server_capabilities:
                    if (client_cap.name == server_cap.name and 
                        client_cap.version == server_cap.version):
                        common_capabilities.append(client_cap)
            
            return common_capabilities
            
        except Exception as e:
            raise ValueError(f"Capability negotiation failed: {str(e)}")
    
    def get_resource(self, resource_id: str) -> Dict[str, Any]:
        """Get a resource from the MCP server.
        
        Args:
            resource_id: ID of the resource to retrieve.
            
        Returns:
            The resource data.
            
        Raises:
            ValueError: If the resource cannot be retrieved.
        """
        return self._send_request("resource/get", {"id": resource_id})
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from the MCP server.
        
        Returns:
            List of available resources.
            
        Raises:
            ValueError: If resources cannot be listed.
        """
        result = self._send_request("resource/list")
        return result.get("resources", [])
    
    def call_tool(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server.
        
        Args:
            tool_id: ID of the tool to call.
            params: Parameters for the tool.
            
        Returns:
            The result of the tool call.
            
        Raises:
            ValueError: If the tool call fails.
        """
        return self._send_request("tool/call", {
            "id": tool_id,
            "params": params
        })
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server.
        
        Returns:
            List of available tools.
            
        Raises:
            ValueError: If tools cannot be listed.
        """
        result = self._send_request("tool/list")
        return result.get("tools", [])
    
    def get_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """Get a prompt template from the MCP server.
        
        Args:
            prompt_id: ID of the prompt to retrieve.
            
        Returns:
            The prompt data.
            
        Raises:
            ValueError: If the prompt cannot be retrieved.
        """
        return self._send_request("prompt/get", {"id": prompt_id})
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts from the MCP server.
        
        Returns:
            List of available prompts.
            
        Raises:
            ValueError: If prompts cannot be listed.
        """
        result = self._send_request("prompt/list")
        return result.get("prompts", [])
    
    def close(self):
        """Close the connection to the MCP server."""
        try:
            self._send_request("shutdown")
        except:
            pass
        finally:
            self.session.close() 