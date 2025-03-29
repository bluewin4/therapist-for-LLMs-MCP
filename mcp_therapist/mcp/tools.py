"""
MCP Tools for the MCP Therapist system.

This module contains tool implementations that expose functionality
to language models through the MCP protocol.
"""

import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import asyncio

from mcp_therapist.mcp.client import MCPTool, MCPClient
from mcp_therapist.models.conversation import Conversation, RutType, InterventionStrategy
from mcp_therapist.core.interventions.manager import InterventionManager
from mcp_therapist.utils.logging import logger
from mcp_therapist.config import settings


class InterventionTools:
    """
    MCP tools for intervention-related operations.
    
    This class provides tools that allow the language model to
    interact with the intervention system, including creating plans,
    evaluating interventions, and getting intervention data.
    """
    
    def __init__(
        self, 
        mcp_client: MCPClient, 
        intervention_manager: InterventionManager
    ):
        """
        Initialize the intervention tools.
        
        Args:
            mcp_client: The MCP client to register tools with
            intervention_manager: The intervention manager to use
        """
        self.mcp_client = mcp_client
        self.intervention_manager = intervention_manager
        self.logger = logger
        
        # Register all tools
        asyncio.create_task(self._register_tools())
    
    async def _register_tools(self):
        """Register all intervention tools with the MCP client."""
        await self.mcp_client.register_tool(
            MCPTool(
                id="get_intervention_statistics",
                name="Get Intervention Statistics",
                description="Get statistics about intervention effectiveness by strategy and rut type",
                function=self.get_intervention_statistics,
                parameters={
                    "type": "object",
                    "properties": {
                        "strategy_type": {
                            "type": "string",
                            "description": "Optional filter for a specific strategy type"
                        },
                        "rut_type": {
                            "type": "string",
                            "description": "Optional filter for a specific rut type"
                        }
                    },
                    "required": []
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "by_strategy": {
                            "type": "object",
                            "description": "Statistics broken down by strategy"
                        },
                        "by_rut_type": {
                            "type": "object", 
                            "description": "Statistics broken down by rut type"
                        },
                        "overall": {
                            "type": "object",
                            "description": "Overall statistics"
                        }
                    }
                }
            )
        )
        
        await self.mcp_client.register_tool(
            MCPTool(
                id="get_intervention_history",
                name="Get Intervention History",
                description="Get the history of interventions for a conversation",
                function=self.get_intervention_history,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "ID of the conversation to get history for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of interventions to return"
                        }
                    },
                    "required": ["conversation_id"]
                },
                return_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "strategy_type": {"type": "string"},
                            "rut_type": {"type": "string"},
                            "timestamp": {"type": "string"},
                            "success": {"type": "boolean"}
                        }
                    }
                }
            )
        )
        
        await self.mcp_client.register_tool(
            MCPTool(
                id="create_intervention_plan",
                name="Create Intervention Plan",
                description="Create a plan for intervening in a conversation",
                function=self.create_intervention_plan,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "ID of the conversation to create a plan for"
                        },
                        "rut_type": {
                            "type": "string",
                            "description": "Type of rut detected",
                            "enum": [rt.value for rt in RutType]
                        },
                        "strategy_type": {
                            "type": "string", 
                            "description": "Strategy to use for intervention",
                            "enum": [st.value for st in InterventionStrategy]
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in the rut detection (0-1)"
                        }
                    },
                    "required": ["conversation_id", "rut_type", "confidence"]
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "conversation_id": {"type": "string"},
                        "rut_type": {"type": "string"},
                        "strategy_type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "timestamp": {"type": "string"}
                    }
                }
            )
        )
        
        await self.mcp_client.register_tool(
            MCPTool(
                id="evaluate_intervention",
                name="Evaluate Intervention",
                description="Evaluate the effectiveness of an intervention",
                function=self.evaluate_intervention, 
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "ID of the conversation"
                        },
                        "intervention_id": {
                            "type": "string",
                            "description": "ID of the intervention to evaluate"
                        }
                    },
                    "required": ["conversation_id", "intervention_id"]
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "metrics": {
                            "type": "object",
                            "description": "Various metrics used for evaluation"
                        },
                        "analysis": {"type": "string"}
                    }
                }
            )
        )
    
    async def get_intervention_statistics(
        self, 
        strategy_type: Optional[str] = None, 
        rut_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about intervention effectiveness.
        
        Args:
            strategy_type: Optional filter by strategy type
            rut_type: Optional filter by rut type
            
        Returns:
            Dictionary of intervention statistics
        """
        self.logger.info(
            f"MCP Tool: Getting intervention statistics (strategy={strategy_type}, rut={rut_type})"
        )
        
        # Get statistics from intervention manager
        stats = self.intervention_manager.get_intervention_statistics()
        
        # Apply filters if specified
        if strategy_type:
            if "by_strategy" in stats and strategy_type in stats["by_strategy"]:
                strategy_stats = stats["by_strategy"][strategy_type]
                stats = {
                    "by_strategy": {strategy_type: strategy_stats},
                    "overall": strategy_stats
                }
            else:
                stats = {"by_strategy": {}, "overall": {"total": 0, "success": 0, "rate": 0}}
        
        if rut_type:
            if "by_rut_type" in stats and rut_type in stats["by_rut_type"]:
                rut_stats = stats["by_rut_type"][rut_type]
                if "by_rut_type" not in stats:
                    stats["by_rut_type"] = {}
                stats["by_rut_type"] = {rut_type: rut_stats}
                if "overall" not in stats:
                    stats["overall"] = rut_stats
        
        return stats
    
    async def get_intervention_history(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the history of interventions for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of interventions to return
            
        Returns:
            List of intervention records
        """
        self.logger.info(f"MCP Tool: Getting intervention history for {conversation_id}")
        
        # Get history from intervention manager
        history = self.intervention_manager.get_intervention_history(conversation_id)
        
        # Apply limit if specified
        if limit and len(history) > limit:
            history = history[-limit:]
        
        return history
    
    async def create_intervention_plan(
        self, 
        conversation_id: str, 
        rut_type: str, 
        confidence: float,
        strategy_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a plan for intervening in a conversation.
        
        Args:
            conversation_id: ID of the conversation
            rut_type: Type of rut detected
            confidence: Confidence in the detection
            strategy_type: Optional strategy to use
            
        Returns:
            Intervention plan
        """
        self.logger.info(
            f"MCP Tool: Creating intervention plan for {conversation_id} "
            f"(rut={rut_type}, confidence={confidence})"
        )
        
        # Convert string types to enums
        try:
            rut_type_enum = RutType(rut_type)
        except ValueError:
            self.logger.error(f"Invalid rut type: {rut_type}")
            raise ValueError(f"Invalid rut type: {rut_type}")
        
        if strategy_type:
            try:
                strategy_type_enum = InterventionStrategy(strategy_type)
            except ValueError:
                self.logger.error(f"Invalid strategy type: {strategy_type}")
                raise ValueError(f"Invalid strategy type: {strategy_type}")
        else:
            strategy_type_enum = None
        
        # Create detection result for the intervention manager
        detection_result = {
            "rut_type": rut_type_enum,
            "confidence": confidence,
            "context": {}
        }
        
        # Create the intervention plan
        # We need to get the conversation first
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        plan = self.intervention_manager.create_intervention_plan(
            conversation=conversation,
            detection_result=detection_result,
            strategy_type=strategy_type_enum
        )
        
        # Convert enums to strings for JSON serialization
        if "rut_type" in plan and isinstance(plan["rut_type"], RutType):
            plan["rut_type"] = plan["rut_type"].value
        
        if "strategy_type" in plan and isinstance(plan["strategy_type"], InterventionStrategy):
            plan["strategy_type"] = plan["strategy_type"].value
        
        return plan
    
    async def evaluate_intervention(
        self, 
        conversation_id: str, 
        intervention_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of an intervention.
        
        Args:
            conversation_id: ID of the conversation
            intervention_id: ID of the intervention to evaluate
            
        Returns:
            Evaluation result
        """
        self.logger.info(
            f"MCP Tool: Evaluating intervention {intervention_id} for {conversation_id}"
        )
        
        # Get the conversation
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Evaluate the intervention
        result = self.intervention_manager.evaluate_intervention_effectiveness(
            conversation=conversation,
            intervention_id=intervention_id
        )
        
        return result
    
    async def _get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Helper method to retrieve a conversation by ID.
        
        This would typically access a database or storage system.
        For now, we'll assume this is implemented elsewhere.
        
        Args:
            conversation_id: ID of the conversation to get
            
        Returns:
            The conversation object or None if not found
        """
        # This is a placeholder that would be replaced with actual implementation
        # In a real system, this would access a database or storage system
        raise NotImplementedError(
            "This method needs to be implemented to access conversation storage"
        )


class ConversationTools:
    """
    MCP tools for conversation-related operations.
    
    This class provides tools that allow the language model to
    analyze conversations, detect ruts, and access conversation data.
    """
    
    def __init__(
        self, 
        mcp_client: MCPClient,
        get_conversation_func: Callable[[str], Conversation] = None
    ):
        """
        Initialize the conversation tools.
        
        Args:
            mcp_client: The MCP client to register tools with
            get_conversation_func: Function to retrieve conversations
        """
        self.mcp_client = mcp_client
        self.get_conversation = get_conversation_func
        self.logger = logger
        
        # Register all tools
        asyncio.create_task(self._register_tools())
    
    async def _register_tools(self):
        """Register all conversation tools with the MCP client."""
        await self.mcp_client.register_tool(
            MCPTool(
                id="analyze_conversation_for_ruts",
                name="Analyze Conversation for Ruts",
                description="Analyze a conversation to detect conversational ruts",
                function=self.analyze_conversation_for_ruts,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "ID of the conversation to analyze"
                        },
                        "window_size": {
                            "type": "integer",
                            "description": "Number of recent messages to analyze"
                        }
                    },
                    "required": ["conversation_id"]
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "rut_detected": {"type": "boolean"},
                        "rut_type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "context": {"type": "object"}
                    }
                }
            )
        )
        
        await self.mcp_client.register_tool(
            MCPTool(
                id="get_conversation_summary",
                name="Get Conversation Summary",
                description="Get a summary of a conversation",
                function=self.get_conversation_summary,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "ID of the conversation to summarize"
                        }
                    },
                    "required": ["conversation_id"]
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "sentiment": {"type": "string"},
                        "message_count": {"type": "integer"}
                    }
                }
            )
        )
        
        await self.mcp_client.register_tool(
            MCPTool(
                id="get_conversation_statistics",
                name="Get Conversation Statistics",
                description="Get statistics about a conversation",
                function=self.get_conversation_statistics,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "ID of the conversation to get statistics for"
                        }
                    },
                    "required": ["conversation_id"]
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "message_count": {"type": "integer"},
                        "user_message_count": {"type": "integer"},
                        "assistant_message_count": {"type": "integer"},
                        "average_user_message_length": {"type": "number"},
                        "average_assistant_message_length": {"type": "number"},
                        "intervention_count": {"type": "integer"}
                    }
                }
            )
        )
    
    async def analyze_conversation_for_ruts(
        self, 
        conversation_id: str, 
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze a conversation to detect conversational ruts.
        
        Args:
            conversation_id: ID of the conversation to analyze
            window_size: Optional number of recent messages to analyze
            
        Returns:
            Detection result
        """
        self.logger.info(
            f"MCP Tool: Analyzing conversation {conversation_id} for ruts"
        )
        
        # Get the conversation
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Limit the window size if specified
        if window_size and window_size > 0 and len(conversation.messages) > window_size:
            # Create a new conversation object with limited messages
            limited_conversation = Conversation(
                id=conversation.id,
                messages=conversation.messages[-window_size:],
                metadata=conversation.metadata
            )
        else:
            limited_conversation = conversation
        
        # Analyze the conversation (placeholder implementation)
        # In a real system, this would use the rut detection system
        result = self._analyze_for_ruts(limited_conversation)
        
        # Format the result
        if result["rut_detected"] and isinstance(result["rut_type"], RutType):
            result["rut_type"] = result["rut_type"].value
        
        return result
    
    async def get_conversation_summary(
        self, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get a summary of a conversation.
        
        Args:
            conversation_id: ID of the conversation to summarize
            
        Returns:
            Conversation summary
        """
        self.logger.info(f"MCP Tool: Getting summary for conversation {conversation_id}")
        
        # Get the conversation
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Generate summary (placeholder implementation)
        # In a real system, this would use a summarization technique
        summary = self._generate_summary(conversation)
        
        return summary
    
    async def get_conversation_statistics(
        self, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics about a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation statistics
        """
        self.logger.info(f"MCP Tool: Getting statistics for conversation {conversation_id}")
        
        # Get the conversation
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Calculate statistics
        stats = self._calculate_statistics(conversation)
        
        return stats
    
    async def _get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Helper method to retrieve a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation to get
            
        Returns:
            The conversation object or None if not found
        """
        if self.get_conversation:
            return await self.get_conversation(conversation_id)
        
        # Placeholder implementation
        raise NotImplementedError(
            "This method needs to be implemented to access conversation storage"
        )
    
    def _analyze_for_ruts(self, conversation: Conversation) -> Dict[str, Any]:
        """
        Placeholder method for analyzing a conversation for ruts.
        
        In a real implementation, this would use the rut detection system.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Detection result
        """
        # Placeholder: In a real implementation, this would use the detector registry
        return {
            "rut_detected": False,
            "rut_type": None,
            "confidence": 0.0,
            "context": {}
        }
    
    def _generate_summary(self, conversation: Conversation) -> Dict[str, Any]:
        """
        Placeholder method for generating a conversation summary.
        
        In a real implementation, this would use a summarization technique.
        
        Args:
            conversation: The conversation to summarize
            
        Returns:
            Conversation summary
        """
        # Placeholder implementation
        return {
            "summary": "Placeholder summary",
            "topics": ["topic1", "topic2"],
            "sentiment": "neutral",
            "message_count": len(conversation.messages)
        }
    
    def _calculate_statistics(self, conversation: Conversation) -> Dict[str, Any]:
        """
        Calculate statistics for a conversation.
        
        Args:
            conversation: The conversation to calculate statistics for
            
        Returns:
            Conversation statistics
        """
        user_messages = [msg for msg in conversation.messages if msg.role == "user"]
        assistant_messages = [msg for msg in conversation.messages if msg.role == "assistant"]
        
        # Count interventions
        intervention_count = 0
        for msg in conversation.messages:
            if msg.metadata and "intervention_id" in msg.metadata:
                intervention_count += 1
        
        # Calculate statistics
        stats = {
            "message_count": len(conversation.messages),
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "average_user_message_length": (
                sum(len(msg.content) for msg in user_messages) / len(user_messages)
                if user_messages else 0
            ),
            "average_assistant_message_length": (
                sum(len(msg.content) for msg in assistant_messages) / len(assistant_messages)
                if assistant_messages else 0
            ),
            "intervention_count": intervention_count
        }
        
        return stats 