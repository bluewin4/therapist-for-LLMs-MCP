"""
Configuration settings for the MCP Therapist system.

Settings can be overridden by environment variables.
"""

import os
from typing import Dict, List, Optional, Union, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPTherapistSettings(BaseSettings):
    """Configuration settings for the MCP Therapist system."""
    
    # General settings
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Conversation settings
    DEFAULT_WINDOW_SIZE: int = Field(default=5, description="Default conversation window size for analysis")
    MAX_CONVERSATION_HISTORY: int = Field(default=100, description="Maximum number of messages to store per conversation")
    
    # Rut detection settings
    REPETITION_THRESHOLD: float = Field(default=0.7, description="Similarity threshold for repetition detection")
    STAGNATION_THRESHOLD: float = Field(default=0.85, description="Semantic similarity threshold for stagnation detection")
    REFUSAL_KEYWORDS: List[str] = Field(
        default=["cannot", "unable", "not allowed", "not permitted", "policy", "I'm sorry"],
        description="Keywords for detecting refusal patterns"
    )
    
    # Intervention settings
    INTERVENTION_COOLDOWN: int = Field(default=3, description="Minimum number of turns between interventions")
    MAX_INTERVENTIONS_PER_CONVERSATION: int = Field(default=5, description="Maximum number of interventions per conversation")
    MIN_INTERVENTION_CONFIDENCE: float = Field(default=0.7, description="Minimum confidence for triggering intervention")
    
    # Intervention injection settings
    DEFAULT_INJECTION_METHOD: str = Field(
        default="SELF_REFLECTION", 
        description="Default method for injecting interventions (DIRECT, SELF_REFLECTION, PREPEND, INLINE, METADATA_ONLY)"
    )
    USE_STRATEGY_SPECIFIC_INJECTION: bool = Field(
        default=True,
        description="Whether to use different injection methods for different strategies"
    )
    
    # Intervention evaluation settings
    INTERVENTION_SUCCESS_THRESHOLD: float = Field(
        default=0.6,
        description="Threshold score for considering an intervention successful"
    )
    MIN_MESSAGES_TO_EVALUATE: int = Field(
        default=2,
        description="Minimum number of messages after intervention before evaluation"
    )
    EVALUATION_WINDOW: int = Field(
        default=5,
        description="Number of messages before/after intervention to analyze for evaluation"
    )
    TRACK_INTERVENTION_HISTORY: bool = Field(
        default=True,
        description="Whether to track intervention history for effectiveness analysis"
    )
    
    # MCP settings
    MCP_ENABLED: bool = Field(default=False, description="Enable MCP integration")
    MCP_SERVER_URL: Optional[str] = Field(default=None, description="MCP server URL")
    
    # Model settings (for prompt crafting)
    SECONDARY_LLM_ENABLED: bool = Field(default=False, description="Enable secondary LLM for prompt crafting")
    SECONDARY_LLM_API_KEY: Optional[str] = Field(default=None, description="API key for secondary LLM")
    SECONDARY_LLM_MODEL: str = Field(default="gpt-3.5-turbo", description="Model name for secondary LLM")
    
    # Embeddings settings
    EMBEDDINGS_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for sentence-transformers"
    )
    USE_GPU_FOR_EMBEDDINGS: bool = Field(
        default=True,
        description="Whether to use GPU for embeddings generation"
    )
    EMBEDDINGS_CACHE_SIZE: int = Field(
        default=1000,
        description="Maximum number of embeddings to cache"
    )
    EMBEDDINGS_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for processing multiple embeddings"
    )

    # Sentiment analysis settings
    SENTIMENT_THRESHOLD_NEGATIVE: float = Field(
        default=0.3,
        description="Threshold for detecting negative sentiment"
    )
    SENTIMENT_THRESHOLD_POSITIVE: float = Field(
        default=0.6,
        description="Threshold for detecting positive sentiment"
    )
    USE_SENTIMENT_ANALYSIS: bool = Field(
        default=True,
        description="Whether to use sentiment analysis"
    )

    # Topic detection settings
    TOPIC_FIXATION_THRESHOLD: float = Field(
        default=0.8,
        description="Threshold for topic similarity to detect fixation"
    )
    TOPIC_FIXATION_MIN_MESSAGES: int = Field(
        default=3,
        description="Minimum number of messages on same topic to detect fixation"
    )
    USE_ADVANCED_TOPIC_DETECTION: bool = Field(
        default=True,
        description="Whether to use advanced topic detection with TF-IDF"
    )

    # Contradiction detection settings
    CONTRADICTION_THRESHOLD: float = Field(
        default=0.7,
        description="Threshold for detecting contradictions"
    )
    USE_CONTRADICTION_DETECTION: bool = Field(
        default=True,
        description="Whether to use contradiction detection"
    )

    # Adaptive strategy selection settings
    STRATEGY_LEARNING_RATE: float = Field(default=0.1, description="Learning rate for strategy selection")
    STRATEGY_EXPLORATION_FACTOR: float = Field(default=0.2, description="Exploration factor for strategy selection")
    STRATEGY_SELECTION_ALGORITHM: str = Field(default="ucb", description="Algorithm for strategy selection")

    model_config = SettingsConfigDict(
        env_prefix="MCP_THERAPIST_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Create an instance of the settings
settings = MCPTherapistSettings()

# =========================================================================
# MCP INTEGRATION SETTINGS
# =========================================================================

# General MCP settings
MCP_ENABLED = True
MCP_SERVER_URL = "http://localhost:8000/mcp"
MCP_API_KEY = None  # Set to None to use default config or specify a key
MCP_CONNECTION_TIMEOUT = 10  # Seconds
MCP_REQUEST_TIMEOUT = 30  # Seconds

# Resource settings
MCP_DEFAULT_RESOURCE_VISIBILITY = "model_only"  # model_only, user_only, user_and_model
MCP_RESOURCE_EXPIRATION = 3600  # Seconds, 0 for no expiration
MCP_MAX_RESOURCE_SIZE = 1024 * 1024  # 1MB

# Tool settings
MCP_TOOL_TIMEOUT = 60  # Seconds
MCP_TOOL_CONCURRENCY = 3  # Max number of concurrent tool invocations
MCP_ALLOWED_TOOLS = ["*"]  # "*" for all, or list specific tools
MCP_ENABLE_DANGEROUS_TOOLS = False  # Set to True to enable potentially dangerous tools

# Prompt settings
MCP_DEFAULT_PROMPT_PARAMS = {
    "type": "object",
    "properties": {}
}
MCP_PROMPT_TEMPLATE_SYNTAX = "string_format"  # string_format, jinja2, handlebars

# Sampling settings
SAMPLING_ENABLED = True
SAMPLING_DEFAULT_TEMPERATURE = 0.7
SAMPLING_DEFAULT_TOP_P = 0.95
SAMPLING_DEFAULT_MAX_TOKENS = 800
SAMPLING_DEFAULT_MODEL = "gpt-4"
SAMPLING_REFLECTION_MODEL = "gpt-4"
SAMPLING_DEFAULT_SYSTEM_PROMPT = """
You are a thoughtful, empathetic therapeutic assistant based on best practices in psychology.
Your goal is to help users explore their thoughts, feelings, and behaviors in a safe and
supportive environment. You use therapeutic techniques adapted from various evidence-based
approaches including CBT, ACT, humanistic therapy, and motivational interviewing.

Remember:
1. Focus on the user's experience and validate their feelings
2. Ask open-ended questions that promote reflection
3. Look for patterns and gently highlight them when relevant
4. Empower the user to develop their own insights rather than directing them
5. Maintain appropriate boundaries and ethical standards

You are NOT a replacement for professional therapy. For serious mental health concerns,
always encourage seeking professional help.
"""

# Connection settings
MCP_RECONNECT_ATTEMPTS = 3
MCP_RECONNECT_BACKOFF = 2.0  # Seconds, multiplied by attempt number
MCP_KEEP_ALIVE_INTERVAL = 30  # Seconds 