"""
MCP Sampling for the MCP Therapist system.

This module contains functionality for LLM sampling through the MCP protocol,
allowing for recursive LLM interactions and advanced generation capabilities.
"""

import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import json
import asyncio

from mcp_therapist.mcp.client import MCPClient
from mcp_therapist.utils.logging import logger
from mcp_therapist.config import settings


class SamplingParameters:
    """
    Parameters for LLM sampling through MCP.
    
    This class encapsulates the parameters used for sampling text
    from language models via the MCP protocol.
    """
    
    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 800,
        stop_sequences: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        model: Optional[str] = None
    ):
        """
        Initialize sampling parameters.
        
        Args:
            temperature: Controls randomness (higher = more random)
            top_p: Controls diversity via nucleus sampling
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that stop generation
            presence_penalty: Penalizes repeated tokens
            frequency_penalty: Penalizes frequent tokens
            model: Specific model to use for sampling
        """
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences or []
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.model = model or settings.SAMPLING_DEFAULT_MODEL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "model": self.model
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SamplingParameters':
        """Create parameters from a dictionary."""
        return cls(
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            max_tokens=data.get("max_tokens", 800),
            stop_sequences=data.get("stop_sequences", []),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            model=data.get("model")
        )


class SamplingContext:
    """
    Context for LLM sampling through MCP.
    
    This class encapsulates the context used for sampling text
    from language models via the MCP protocol.
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        resource_ids: Optional[List[str]] = None,
        user_prompt: Optional[str] = None
    ):
        """
        Initialize sampling context.
        
        Args:
            system_prompt: System prompt for the language model
            conversation_history: Previous conversation messages
            resource_ids: IDs of MCP resources to include
            user_prompt: The user prompt for the current sampling
        """
        self.system_prompt = system_prompt
        self.conversation_history = conversation_history or []
        self.resource_ids = resource_ids or []
        self.user_prompt = user_prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to a dictionary."""
        return {
            "system_prompt": self.system_prompt,
            "conversation_history": self.conversation_history,
            "resource_ids": self.resource_ids,
            "user_prompt": self.user_prompt
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SamplingContext':
        """Create context from a dictionary."""
        return cls(
            system_prompt=data.get("system_prompt"),
            conversation_history=data.get("conversation_history", []),
            resource_ids=data.get("resource_ids", []),
            user_prompt=data.get("user_prompt")
        )


class SamplingResult:
    """
    Result of LLM sampling through MCP.
    
    This class encapsulates the result of sampling text
    from language models via the MCP protocol.
    """
    
    def __init__(
        self,
        text: str,
        finish_reason: str,
        usage: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize sampling result.
        
        Args:
            text: The generated text
            finish_reason: Reason for finishing (e.g. "stop", "length")
            usage: Token usage information
            metadata: Additional metadata about the sampling
        """
        self.text = text
        self.finish_reason = finish_reason
        self.usage = usage
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary."""
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SamplingResult':
        """Create result from a dictionary."""
        return cls(
            text=data.get("text", ""),
            finish_reason=data.get("finish_reason", "unknown"),
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            metadata=data.get("metadata", {})
        )


class MCPSamplingManager:
    """
    Manager for LLM sampling through MCP.
    
    This class provides methods for sampling text from language
    models via the MCP protocol, with support for recursive
    interactions and advanced generation capabilities.
    """
    
    def __init__(self, mcp_client: MCPClient):
        """
        Initialize the sampling manager.
        
        Args:
            mcp_client: The MCP client to use for sampling
        """
        self.mcp_client = mcp_client
        self.logger = logger
        
        # Default sampling parameters
        self.default_parameters = SamplingParameters()
        
        # Default therapy system prompt (can be customized)
        self.default_system_prompt = settings.SAMPLING_DEFAULT_SYSTEM_PROMPT
    
    async def sample_text(
        self, 
        context: Union[SamplingContext, Dict[str, Any]],
        parameters: Optional[Union[SamplingParameters, Dict[str, Any]]] = None
    ) -> SamplingResult:
        """
        Sample text from a language model.
        
        Args:
            context: The sampling context
            parameters: Optional sampling parameters
            
        Returns:
            The sampling result
        """
        # Convert dictionaries to proper objects if needed
        if isinstance(context, dict):
            context = SamplingContext.from_dict(context)
        
        if parameters is None:
            parameters = self.default_parameters
        elif isinstance(parameters, dict):
            parameters = SamplingParameters.from_dict(parameters)
        
        # If no system prompt is provided, use the default
        if not context.system_prompt:
            context.system_prompt = self.default_system_prompt
        
        self.logger.info(f"Sampling text with {parameters.model} model")
        
        # Use the MCP client to sample text
        result = await self.mcp_client.sample_text(
            context=context.to_dict(),
            parameters=parameters.to_dict()
        )
        
        # Convert the result to a SamplingResult object
        if isinstance(result, dict):
            return SamplingResult.from_dict(result)
        
        return result
    
    async def analyze_with_llm(
        self,
        prompt: str,
        resource_ids: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        parameters: Optional[SamplingParameters] = None
    ) -> str:
        """
        Analyze text with a language model.
        
        This is a convenience method for simple analyses that
        don't require conversation history.
        
        Args:
            prompt: The analysis prompt
            resource_ids: Optional MCP resource IDs to include
            system_prompt: Optional system prompt
            parameters: Optional sampling parameters
            
        Returns:
            The analysis result text
        """
        context = SamplingContext(
            system_prompt=system_prompt or "You are a helpful assistant that analyzes text.",
            resource_ids=resource_ids,
            user_prompt=prompt
        )
        
        # Use more focused parameters for analysis
        if parameters is None:
            parameters = SamplingParameters(
                temperature=0.3,  # Lower temperature for more focused analysis
                max_tokens=500
            )
        
        result = await self.sample_text(context, parameters)
        return result.text
    
    async def stream_sample(
        self, 
        context: SamplingContext,
        parameters: Optional[SamplingParameters] = None,
        callback: Callable[[str], None] = None
    ) -> SamplingResult:
        """
        Stream text sampling from a language model.
        
        Args:
            context: The sampling context
            parameters: Optional sampling parameters
            callback: Function to call with each chunk of text
            
        Returns:
            The complete sampling result
        """
        if parameters is None:
            parameters = self.default_parameters
        
        # If no system prompt is provided, use the default
        if not context.system_prompt:
            context.system_prompt = self.default_system_prompt
        
        self.logger.info(f"Streaming text sampling with {parameters.model} model")
        
        # Use the MCP client to stream text
        full_text = ""
        
        # This would be replaced with actual streaming implementation
        # We're simulating it here with a simple call
        result = await self.mcp_client.sample_text(
            context=context.to_dict(),
            parameters=parameters.to_dict()
        )
        
        if callback:
            callback(result.text)
        
        return result
    
    async def multi_step_reasoning(
        self,
        question: str,
        steps: int = 3,
        resource_ids: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform multi-step reasoning with a language model.
        
        This method uses recursive sampling to break down a complex
        question into steps, then combines the results.
        
        Args:
            question: The question to reason about
            steps: Number of reasoning steps
            resource_ids: Optional MCP resource IDs to include
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with the reasoning steps and final answer
        """
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that breaks down complex questions
            into steps, thinks through each step carefully, and then provides a final answer."""
        
        # Step 1: Break down the question
        breakdown_context = SamplingContext(
            system_prompt=system_prompt,
            resource_ids=resource_ids,
            user_prompt=f"""
            I need to answer the following question: "{question}"
            
            Please break this down into {steps} sequential reasoning steps that will help me
            arrive at a comprehensive answer. For each step, explain what information I need
            to consider and why it's relevant.
            
            Format the steps as:
            Step 1: [brief description]
            Step 2: [brief description]
            ...
            """
        )
        
        breakdown_parameters = SamplingParameters(
            temperature=0.3,
            max_tokens=500
        )
        
        breakdown_result = await self.sample_text(breakdown_context, breakdown_parameters)
        step_breakdown = breakdown_result.text
        
        # Step 2: Execute each reasoning step
        step_results = []
        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": step_breakdown}
        ]
        
        for i in range(1, steps + 1):
            step_context = SamplingContext(
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                resource_ids=resource_ids,
                user_prompt=f"Let's execute Step {i} from your breakdown. Think through this step in detail."
            )
            
            step_parameters = SamplingParameters(
                temperature=0.5,
                max_tokens=800
            )
            
            step_result = await self.sample_text(step_context, step_parameters)
            step_results.append(step_result.text)
            
            # Add this step to the conversation history
            conversation_history.append({"role": "user", "content": f"Let's execute Step {i} from your breakdown."})
            conversation_history.append({"role": "assistant", "content": step_result.text})
        
        # Step 3: Generate final answer
        final_context = SamplingContext(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            resource_ids=resource_ids,
            user_prompt=f"""
            Based on the reasoning steps we've worked through, please provide a final,
            comprehensive answer to the original question: "{question}"
            """
        )
        
        final_parameters = SamplingParameters(
            temperature=0.7,
            max_tokens=1000
        )
        
        final_result = await self.sample_text(final_context, final_parameters)
        
        # Combine all results
        return {
            "question": question,
            "step_breakdown": step_breakdown,
            "step_results": step_results,
            "final_answer": final_result.text,
            "token_usage": final_result.usage
        }
    
    async def create_therapy_reflection(
        self,
        conversation_resource_id: str,
        focus_area: Optional[str] = None,
        parameters: Optional[SamplingParameters] = None
    ) -> str:
        """
        Create a therapy reflection for a conversation.
        
        This method uses the LLM to generate a reflection on a therapy
        conversation, focusing on patterns, insights, and next steps.
        
        Args:
            conversation_resource_id: ID of the conversation resource
            focus_area: Optional specific area to focus on
            parameters: Optional sampling parameters
            
        Returns:
            The reflection text
        """
        system_prompt = """
        You are a thoughtful therapeutic assistant analyzing a conversation.
        Your goal is to provide insightful reflection on the patterns, themes,
        and underlying dynamics in the conversation. Consider both what was said
        and what might be unsaid or implied. Be empathetic, nuanced, and focused
        on understanding rather than judging.
        """
        
        prompt = """
        Please analyze the provided conversation and create a therapeutic reflection.
        
        Focus on:
        1. Key themes and patterns in the conversation
        2. Emotional undercurrents and unspoken elements
        3. Potential areas for growth or exploration
        4. Therapeutic approaches that might be beneficial
        """
        
        if focus_area:
            prompt += f"\n\nPlease pay special attention to: {focus_area}"
        
        context = SamplingContext(
            system_prompt=system_prompt,
            resource_ids=[conversation_resource_id],
            user_prompt=prompt
        )
        
        if parameters is None:
            parameters = SamplingParameters(
                temperature=0.7,
                max_tokens=1000,
                model=settings.SAMPLING_REFLECTION_MODEL
            )
        
        result = await self.sample_text(context, parameters)
        return result.text
    
    async def get_sampling_capabilities(self) -> Dict[str, Any]:
        """
        Get the sampling capabilities of the MCP server.
        
        Returns:
            Dictionary of sampling capabilities
        """
        # This would check what capabilities are supported by the MCP server
        # For now, we'll return a placeholder
        capabilities = await self.mcp_client.get_capabilities()
        
        sampling_capabilities = {
            "models": capabilities.get("models", ["gpt-3.5-turbo", "gpt-4"]),
            "streaming": capabilities.get("streaming", False),
            "max_tokens": capabilities.get("max_tokens", 4000),
            "features": capabilities.get("features", ["basic_sampling"])
        }
        
        return sampling_capabilities 