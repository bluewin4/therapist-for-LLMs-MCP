"""
Configuration settings for the MCP Therapist system.

Settings can be overridden by environment variables.
"""

import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
from dataclasses import dataclass, field, asdict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


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


@dataclass
class LoggingSettings:
    """Logging settings."""
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console: bool = True


@dataclass
class EmbeddingSettings:
    """Embedding settings."""
    use_sentence_transformers: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    similarity_threshold: float = 0.75
    oldest_message_weight: float = 0.5  # Weight for oldest messages
    batch_size: int = 32  # Batch size for embedding generation


@dataclass
class DetectorSettings:
    """Detector settings."""
    min_confidence: float = 0.6
    repetition_threshold: int = 3
    min_message_count: int = 3
    patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "refusal": ["cannot", "unable", "not allowed", "not permitted", "policy", "I'm sorry"],
        "stagnation": ["um", "uh", "hmm", "let me think", "I don't know"],
        "repetition": ["as I said", "as mentioned", "like I said", "as previously stated"],
    })
    enabled_detectors: List[str] = field(default_factory=lambda: ["repetition", "stagnation", "refusal"])
    similarity_threshold: float = 0.85


@dataclass
class InterventionSettings:
    """Intervention settings."""
    min_confidence: float = 0.7
    max_interventions_per_hour: int = 5
    cooldown_minutes: int = 10
    default_injection_method: str = "DIRECT"
    default_prompt_template: str = "default_intervention"
    success_threshold: float = 0.6
    evaluation_window_messages: int = 3
    evaluation_timeout_minutes: int = 15


@dataclass
class MCPSettings:
    """Model Context Protocol settings."""
    enabled: bool = True
    auto_register_resources: bool = True
    resource_ttl_seconds: int = 600
    prompt_template_dir: str = "prompts"
    tools_enabled: bool = True
    server_url: str = "http://localhost:8000"
    api_key: Optional[str] = None


@dataclass
class PerformanceSettings:
    """Performance optimization settings."""
    # Caching
    enable_caching: bool = True
    embedding_cache_size: int = 10000  # Number of entries to keep in embedding cache
    embedding_cache_ttl: int = 86400   # Time to live for embedding cache in seconds (24h)
    resource_cache_size: int = 1000    # Number of entries to keep in resource cache
    resource_cache_ttl: int = 3600     # Time to live for resource cache in seconds (1h)
    prompt_cache_size: int = 500       # Number of entries to keep in prompt cache
    prompt_cache_ttl: int = 1800       # Time to live for prompt cache in seconds (30m)
    
    # Parallelization
    max_workers: int = 8               # Maximum number of worker threads
    parallel_detection: bool = True    # Run detectors in parallel
    parallel_batch_processing: bool = True  # Process batch operations in parallel
    detector_batch_size: int = 5       # Batch size for detector parallel processing
    conversation_batch_size: int = 3   # Batch size for conversation processing
    
    # Profiling
    enable_profiling: bool = False     # Enable performance profiling
    profile_output_file: Optional[str] = None  # File to write profile data
    profile_section_threshold_ms: int = 50  # Log sections taking longer than this
    
    # Memory management
    gc_after_large_operations: bool = True  # Force garbage collection after large operations
    memory_warning_threshold_mb: int = 1000  # Log warning when memory exceeds this value


@dataclass
class Settings:
    """Application settings."""
    debug: bool = False
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    detectors: DetectorSettings = field(default_factory=DetectorSettings)
    interventions: InterventionSettings = field(default_factory=InterventionSettings)
    mcp: MCPSettings = field(default_factory=MCPSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update settings from dictionary.
        
        Args:
            data: Dictionary with settings
        """
        # Update top-level settings
        for key, value in data.items():
            if hasattr(self, key):
                if key == "logging":
                    self.logging = LoggingSettings(**value)
                elif key == "embeddings":
                    self.embeddings = EmbeddingSettings(**value)
                elif key == "detectors":
                    self.detectors = DetectorSettings(**value)
                elif key == "interventions":
                    self.interventions = InterventionSettings(**value)
                elif key == "mcp":
                    self.mcp = MCPSettings(**value)
                elif key == "performance":
                    self.performance = PerformanceSettings(**value)
                else:
                    setattr(self, key, value)
    
    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load settings from a JSON file.
        
        Args:
            file_path: Path to settings file
        """
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
            self.from_dict(data)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save settings to a JSON file.
        
        Args:
            file_path: Path to settings file
        """
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Create global settings instance
settings = Settings()

# Load settings from environment variables
if os.environ.get("MCP_THERAPIST_SETTINGS"):
    settings.load_from_file(os.environ["MCP_THERAPIST_SETTINGS"])

# Load local settings if available
local_settings_path = BASE_DIR / "config" / "local_settings.json"
if local_settings_path.exists():
    settings.load_from_file(local_settings_path) 