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
            mcp_client: The MCP client to register prompts with
        """
        self.mcp_client = mcp_client
        self.logger = logger
        self.prompts = {}
        
        # Register default prompts
        self._register_default_prompts()
    
    def _register_default_prompts(self):
        """Register the default set of therapy prompts."""
        # Intervention prompts
        self._register_intervention_prompts()
        
        # Rut detection prompts
        self._register_rut_detection_prompts()
        
        # Therapy session prompts
        self._register_therapy_session_prompts()
    
    def _register_intervention_prompts(self):
        """Register prompts related to interventions."""
        # Register prompt for direct interventions
        self._register_prompt(
            MCPPrompt(
                id="direct_intervention",
                name="Direct Intervention",
                description="Template for direct interventions in conversations",
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
                }
            )
        )
        
        # Register prompt for self-reflection interventions
        self._register_prompt(
            MCPPrompt(
                id="self_reflection_intervention",
                name="Self-Reflection Intervention",
                description="Template for self-reflection prompts for assistants",
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
                }
            )
        )
        
        # Register prompt for inline interventions
        self._register_prompt(
            MCPPrompt(
                id="inline_intervention",
                name="Inline Intervention",
                description="Template for inline interventions that are inserted into messages",
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
                }
            )
        )
        
        # Register prompt for prepend interventions
        self._register_prompt(
            MCPPrompt(
                id="prepend_intervention",
                name="Prepend Intervention",
                description="Template for interventions that are prepended to messages",
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
                }
            )
        )
    
    def _register_rut_detection_prompts(self):
        """Register prompts related to rut detection."""
        # Register prompt for circular reasoning analysis
        self._register_prompt(
            MCPPrompt(
                id="circular_reasoning_analysis",
                name="Circular Reasoning Analysis",
                description="Template for analyzing circular reasoning patterns",
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
                }
            )
        )
        
        # Register prompt for repetitive question analysis
        self._register_prompt(
            MCPPrompt(
                id="repetitive_question_analysis",
                name="Repetitive Question Analysis",
                description="Template for analyzing repetitive question patterns",
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
                }
            )
        )
        
        # Register prompt for shallow engagement analysis
        self._register_prompt(
            MCPPrompt(
                id="shallow_engagement_analysis",
                name="Shallow Engagement Analysis",
                description="Template for analyzing shallow engagement patterns",
                template="""
                I'm analyzing the following conversation for shallow engagement patterns:
                
                {conversation_excerpt}
                
                I'll evaluate:
                1. Depth of user responses (word count, complexity, substantive content)
                2. Level of personal disclosure or reflection
                3. Willingness to engage with questions or suggestions
                4. Signs of passive or minimal participation
                
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
                }
            )
        )
    
    def _register_therapy_session_prompts(self):
        """Register prompts related to therapy sessions."""
        # Register prompt for session introduction
        self._register_prompt(
            MCPPrompt(
                id="session_introduction",
                name="Therapy Session Introduction",
                description="Template for introducing a therapy session",
                template="""
                Hello{user_name_greeting}! Welcome to our {session_number} session together.
                
                {if_first_session}Before we begin, I want to remind you that I'm here to support you in a safe, confidential space. While I can offer guidance based on therapeutic principles, I'm not a licensed therapist, and our conversations are not a substitute for professional mental health support.{endif_first_session}
                
                {if_continuation}Last time, we discussed {previous_topics}. How have things been since our last conversation?{endif_continuation}
                
                {if_specific_goal}You mentioned wanting to focus on {specific_goal} today. Would you like to start there, or is there something else on your mind?{endif_specific_goal}
                
                {if_no_specific_goal}What would you like to focus on in our conversation today?{endif_no_specific_goal}
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "user_name": {
                            "type": "string",
                            "description": "The user's name (optional)"
                        },
                        "session_number": {
                            "type": "string",
                            "description": "Ordinal number of the session (e.g., 'first', 'second')"
                        },
                        "is_first_session": {
                            "type": "boolean",
                            "description": "Whether this is the first session"
                        },
                        "is_continuation": {
                            "type": "boolean",
                            "description": "Whether this is a continuation of previous sessions"
                        },
                        "previous_topics": {
                            "type": "string",
                            "description": "Summary of topics from previous sessions"
                        },
                        "specific_goal": {
                            "type": "string",
                            "description": "Specific goal or topic for this session (optional)"
                        }
                    },
                    "required": ["session_number"]
                }
            )
        )
        
        # Register prompt for session conclusion
        self._register_prompt(
            MCPPrompt(
                id="session_conclusion",
                name="Therapy Session Conclusion",
                description="Template for concluding a therapy session",
                template="""
                As we wrap up our conversation for today, I'd like to summarize what we've discussed:
                
                {session_summary}
                
                Key insights:
                {key_insights}
                
                {if_action_items}Some things you might consider before our next conversation:
                {action_items}{endif_action_items}
                
                {if_next_session}I look forward to our next conversation. Feel free to reflect on what we've discussed today in the meantime.{endif_next_session}
                
                Is there anything else you'd like to discuss before we conclude?
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "session_summary": {
                            "type": "string",
                            "description": "Summary of the current session"
                        },
                        "key_insights": {
                            "type": "string",
                            "description": "Key insights or breakthroughs from the session"
                        },
                        "has_action_items": {
                            "type": "boolean",
                            "description": "Whether there are action items to recommend"
                        },
                        "action_items": {
                            "type": "string",
                            "description": "Suggested action items or reflections"
                        },
                        "has_next_session": {
                            "type": "boolean",
                            "description": "Whether a next session is planned"
                        }
                    },
                    "required": ["session_summary"]
                }
            )
        )
        
        # Register prompt for empathetic response
        self._register_prompt(
            MCPPrompt(
                id="empathetic_response",
                name="Empathetic Response",
                description="Template for generating empathetic responses to user disclosures",
                template="""
                I can hear that {emotional_content}. That sounds {emotion_description}.
                
                {validation_statement}
                
                {if_exploration}What aspect of this {exploration_question}{endif_exploration}
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "emotional_content": {
                            "type": "string",
                            "description": "Summary of the emotional content shared"
                        },
                        "emotion_description": {
                            "type": "string",
                            "description": "Description of the emotion (e.g., 'challenging', 'painful')"
                        },
                        "validation_statement": {
                            "type": "string",
                            "description": "Statement validating the user's experience"
                        },
                        "should_explore": {
                            "type": "boolean",
                            "description": "Whether to include an exploration question"
                        },
                        "exploration_question": {
                            "type": "string",
                            "description": "Question to explore the topic further"
                        }
                    },
                    "required": ["emotional_content", "validation_statement"]
                }
            )
        )
    
    def _register_prompt(self, prompt: MCPPrompt):
        """
        Register a prompt in the internal registry and with the MCP client.
        
        Args:
            prompt: The prompt to register
        """
        self.prompts[prompt.id] = prompt
        self.mcp_client.register_prompt(prompt)
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
        self._register_prompt(prompt)
        
        return prompt 