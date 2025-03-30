"""
MCP Prompts for the MCP Therapist system.

This module contains prompt templates that can be used with
the MCP protocol to standardize interactions with language models.
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from mcp_therapist.mcp.client import MCPPrompt, MCPClient
from mcp_therapist.utils.logging import logger
from mcp_therapist.config import settings


class TherapyPromptManager:
    """
    Manager for therapy-related MCP prompts.
    
    This class provides methods for registering, retrieving,
    and rendering prompt templates for therapy interactions.
    """
    
    def __init__(self, mcp_client: MCPClient):
        """
        Initialize the therapy prompt manager.
        
        Args:
            mcp_client: MCP client instance
        """
        self.mcp_client = mcp_client
        self.prompts: Dict[str, MCPPrompt] = {}
        self.logger = logger
    
    async def _register_default_prompts(self):
        """Register the default set of therapy prompts."""
        # Intervention prompts
        await self._register_intervention_prompts()
        
        # Rut detection prompts
        await self._register_rut_detection_prompts()
        
        # Therapy session prompts
        await self._register_therapy_session_prompts()
    
    async def _register_intervention_prompts(self):
        """Register prompts related to interventions."""
        # Register prompt for direct interventions
        await self._register_prompt(
            MCPPrompt(
                id="direct_intervention",
                name="Direct Intervention",
                description="A prompt to directly intervene in the conversation to break a pattern",
                template="""
                I've noticed that our conversation may be getting stuck in a pattern. 
                {intervention_content}
                
                Would you like to explore this topic differently?
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "intervention_content": {
                            "type": "string",
                            "description": "The specific intervention content"
                        }
                    },
                    "required": ["intervention_content"]
                },
                metadata={"type": "intervention", "subtype": "direct"}
            )
        )
        
        # Register prompt for self-reflection interventions
        await self._register_prompt(
            MCPPrompt(
                id="self_reflection_intervention",
                name="Self-Reflection Intervention",
                description="A prompt for the assistant to perform self-reflection on conversation patterns",
                template="""
                [Self-Reflection]
                I need to consider how to address the following pattern in our conversation:
                
                Pattern detected: {rut_type}
                Analysis: {analysis}
                
                Potential strategies:
                1. {strategy_1}
                2. {strategy_2}
                3. {strategy_3}
                
                I'll select the most appropriate strategy based on the conversation context.
                [End Self-Reflection]
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "rut_type": {
                            "type": "string",
                            "description": "The type of conversational rut detected"
                        },
                        "analysis": {
                            "type": "string",
                            "description": "Analysis of the pattern"
                        },
                        "strategy_1": {
                            "type": "string",
                            "description": "First potential strategy"
                        },
                        "strategy_2": {
                            "type": "string",
                            "description": "Second potential strategy"
                        },
                        "strategy_3": {
                            "type": "string",
                            "description": "Third potential strategy"
                        }
                    },
                    "required": ["rut_type", "analysis"]
                },
                metadata={"type": "intervention", "subtype": "self_reflection"}
            )
        )
        
        # Register prompt for inline interventions
        await self._register_prompt(
            MCPPrompt(
                id="inline_intervention",
                name="Inline Intervention",
                description="A prompt to insert an intervention within the existing message",
                template="{message_start} {intervention_content} {message_end}",
                parameters={
                    "type": "object",
                    "properties": {
                        "message_start": {
                            "type": "string",
                            "description": "The start of the original message"
                        },
                        "intervention_content": {
                            "type": "string",
                            "description": "The intervention content to insert"
                        },
                        "message_end": {
                            "type": "string",
                            "description": "The end of the original message"
                        }
                    },
                    "required": ["message_start", "intervention_content", "message_end"]
                },
                metadata={"type": "intervention", "subtype": "inline"}
            )
        )
        
        # Register prompt for prepend interventions
        await self._register_prompt(
            MCPPrompt(
                id="prepend_intervention",
                name="Prepend Intervention",
                description="A prompt to add intervention content before the original message",
                template="{intervention_content}\n\n{original_message}",
                parameters={
                    "type": "object",
                    "properties": {
                        "intervention_content": {
                            "type": "string",
                            "description": "The intervention content to prepend"
                        },
                        "original_message": {
                            "type": "string",
                            "description": "The original message content"
                        }
                    },
                    "required": ["intervention_content", "original_message"]
                },
                metadata={"type": "intervention", "subtype": "prepend"}
            )
        )
    
    async def _register_rut_detection_prompts(self):
        """Register prompts related to rut detection."""
        # Register prompt for circular reasoning analysis
        await self._register_prompt(
            MCPPrompt(
                id="circular_reasoning_analysis",
                name="Circular Reasoning Analysis",
                description="A prompt to analyze conversation for circular reasoning patterns",
                template="""
                I'm analyzing the following conversation for circular reasoning patterns:
                
                {conversation_excerpt}
                
                I'll identify any instances where:
                1. The user is repeating the same arguments without progress
                2. The conversation is looping back to the same points
                3. Premises depend on the conclusion they're trying to prove
                
                Analysis:
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_excerpt": {
                            "type": "string",
                            "description": "Excerpt from the conversation to analyze"
                        }
                    },
                    "required": ["conversation_excerpt"]
                },
                metadata={"type": "analysis", "subtype": "circular_reasoning"}
            )
        )
        
        # Register prompt for repetitive question analysis
        await self._register_prompt(
            MCPPrompt(
                id="repetitive_question_analysis",
                name="Repetitive Question Analysis",
                description="A prompt to analyze conversation for repetitive questioning patterns",
                template="""
                I'm analyzing the following conversation for repetitive questioning patterns:
                
                {conversation_excerpt}
                
                I'll identify:
                1. Questions that have been asked multiple times in similar forms
                2. Questions where the answer has already been provided
                3. Patterns of asking the same question with slight variations
                
                Analysis:
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_excerpt": {
                            "type": "string",
                            "description": "Excerpt from the conversation to analyze"
                        }
                    },
                    "required": ["conversation_excerpt"]
                },
                metadata={"type": "analysis", "subtype": "repetitive_question"}
            )
        )
        
        # Register prompt for shallow engagement analysis
        await self._register_prompt(
            MCPPrompt(
                id="shallow_engagement_analysis",
                name="Shallow Engagement Analysis",
                description="A prompt to analyze conversation for shallow engagement patterns",
                template="""
                I'm analyzing the following conversation for shallow engagement patterns:
                
                {conversation_excerpt}
                
                I'll identify:
                1. Responses that avoid addressing key points
                2. Generic or vague answers that don't advance discussion
                3. Patterns of deflection or minimal engagement
                
                Analysis:
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_excerpt": {
                            "type": "string",
                            "description": "Excerpt from the conversation to analyze"
                        }
                    },
                    "required": ["conversation_excerpt"]
                },
                metadata={"type": "analysis", "subtype": "shallow_engagement"}
            )
        )
    
    async def _register_therapy_session_prompts(self):
        """Register prompts related to therapy sessions."""
        # Register prompt for session introduction
        await self._register_prompt(
            MCPPrompt(
                id="session_introduction",
                name="Therapy Session Introduction",
                description="A prompt for introducing a new therapy session",
                template="""
                Welcome to our session today. I'm here to support you in exploring your thoughts and feelings.
                
                {custom_welcome_message}
                
                What would you like to talk about today?
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "custom_welcome_message": {
                            "type": "string",
                            "description": "Customized welcome message based on user history"
                        }
                    },
                    "required": []
                },
                metadata={"type": "session", "subtype": "introduction"}
            )
        )
        
        # Register prompt for session conclusion
        await self._register_prompt(
            MCPPrompt(
                id="session_conclusion",
                name="Therapy Session Conclusion",
                description="A prompt for concluding a therapy session",
                template="""
                We're coming to the end of our session. 
                
                {session_summary}
                
                {next_steps}
                
                Is there anything else you'd like to address before we conclude?
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "session_summary": {
                            "type": "string",
                            "description": "Brief summary of the session"
                        },
                        "next_steps": {
                            "type": "string",
                            "description": "Suggested next steps or focus areas"
                        }
                    },
                    "required": ["session_summary"]
                },
                metadata={"type": "session", "subtype": "conclusion"}
            )
        )
        
        # Register prompt for empathetic response
        await self._register_prompt(
            MCPPrompt(
                id="empathetic_response",
                name="Empathetic Response",
                description="A prompt for generating an empathetic response to emotional content",
                template="""
                I can sense that {emotional_content}. That sounds really challenging.
                
                {validation_statement}
                
                Would you like to explore this feeling more deeply?
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "emotional_content": {
                            "type": "string",
                            "description": "Description of the emotional state detected"
                        },
                        "validation_statement": {
                            "type": "string",
                            "description": "Statement validating the user's experience"
                        }
                    },
                    "required": ["emotional_content", "validation_statement"]
                },
                metadata={"type": "response", "subtype": "empathetic"}
            )
        )
    
    async def _register_prompt(self, prompt: MCPPrompt):
        """
        Register a prompt in the internal registry and with the MCP client.
        
        Args:
            prompt: The prompt to register
        """
        self.prompts[prompt.id] = prompt
        await self.mcp_client.register_prompt(prompt)
        self.logger.info(f"Registered prompt: {prompt.id}")
    
    async def render_prompt(
        self, 
        prompt_id: str, 
        parameters: Dict[str, Any]
    ) -> str:
        """
        Render a prompt with the given parameters.
        
        Args:
            prompt_id: ID of the prompt to render
            parameters: Parameters for the prompt template
            
        Returns:
            The rendered prompt text
        """
        # This will use the MCP client to render the prompt
        rendered_text = await self.mcp_client.render_prompt(prompt_id, parameters)
        self.logger.info(f"Rendered prompt: {prompt_id}")
        
        return rendered_text
    
    async def get_prompt(self, prompt_id: str) -> Optional[MCPPrompt]:
        """
        Get a prompt by ID.
        
        Args:
            prompt_id: ID of the prompt to get
            
        Returns:
            The prompt or None if not found
        """
        if prompt_id in self.prompts:
            return self.prompts[prompt_id]
        
        # Try to get from MCP client
        prompt = await self.mcp_client.get_prompt(prompt_id)
        if prompt:
            self.prompts[prompt_id] = prompt
        
        return prompt
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """
        List all registered prompts.
        
        Returns:
            List of prompt metadata
        """
        prompts = []
        for prompt_id, prompt in self.prompts.items():
            prompts.append({
                "id": prompt.id,
                "name": prompt.name,
                "description": prompt.description
            })
        
        return prompts
    
    async def create_prompt(
        self, 
        id: str,
        name: str,
        description: str,
        template: str,
        parameters: Dict[str, Any]
    ) -> MCPPrompt:
        """
        Create a new prompt and register it.
        
        Args:
            id: ID for the new prompt
            name: Name of the prompt
            description: Description of the prompt
            template: The prompt template text
            parameters: JSON schema for the template parameters
            
        Returns:
            The created prompt
        """
        # Create the prompt
        prompt = MCPPrompt(
            id=id,
            name=name,
            description=description,
            template=template,
            parameters=parameters
        )
        
        # Register the prompt
        await self._register_prompt(prompt)
        
        return prompt 