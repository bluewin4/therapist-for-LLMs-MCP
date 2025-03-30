# MCP Therapist Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [Detectors](#detectors)
7. [Intervention System](#intervention-system)
8. [MCP Integration](#mcp-integration)
9. [Performance Optimization](#performance-optimization)
10. [API Reference](#api-reference)
11. [Testing](#testing)
12. [Deployment](#deployment)
13. [Troubleshooting](#troubleshooting)
14. [FAQ](#faq)

## Introduction

MCP Therapist is a system designed to monitor interactions with Language Models (LLMs), detect when the conversation falls into unproductive patterns ("ruts"), and apply interventions to guide the conversation back to a more productive state.

The system draws inspiration from therapeutic techniques, providing "infoblessing" prompts that help LLMs overcome limitations without making users aware of the intervention process. It integrates with the Model Context Protocol (MCP) to enhance its capabilities.

### Key Features

- Conversation monitoring and analysis
- Multiple rut detection methods (repetition, stagnation, topic fixation, etc.)
- Strategic intervention selection based on rut type
- Multiple intervention injection methods
- Intervention effectiveness evaluation
- MCP integration for enhanced context sharing
- Performance optimization with caching and parallel processing

## Architecture

MCP Therapist follows a modular architecture organized around these main components:

```
┌─────────────────┐      ┌───────────────┐      ┌───────────────────┐
│ Conversation    │      │ Detection     │      │ Intervention      │
│ Monitoring      │─────▶│ Pipeline      │─────▶│ Management        │
└─────────────────┘      └───────────────┘      └───────────────────┘
        │                                                │
        │                                                │
        ▼                                                ▼
┌─────────────────┐                            ┌───────────────────┐
│ Conversation    │                            │ Evaluation        │
│ History         │◀───────────────────────────│ System            │
└─────────────────┘                            └───────────────────┘
```

- **Conversation Monitoring**: Tracks and stores the conversation between user and LLM
- **Detection Pipeline**: Analyzes conversation to identify problematic patterns
- **Intervention Management**: Selects and applies appropriate interventions
- **Evaluation System**: Measures the effectiveness of interventions
- **MCP Integration**: Connects system to the Model Context Protocol ecosystem

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: GPU for faster embedding generation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp_therapist.git
cd mcp_therapist

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Development Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Configuration

MCP Therapist uses a combination of environment variables and configuration files for settings:

### Environment Variables

Core environment variables are loaded from `.env` files:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# MCP Configuration
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your_mcp_api_key_if_required

# Performance Settings
ENABLE_PROFILING=false
ENABLE_CACHING=true
PARALLEL_PROCESSING=true
```

### Configuration Settings

More detailed settings are available in `mcp_therapist/config/settings.py`:

- **Logging settings**: Configure log levels, formats, and output options
- **Embedding settings**: Model selection, dimension, and thresholds
- **Detector settings**: Confidence thresholds and pattern definitions
- **Intervention settings**: Cooldown periods, evaluation windows, injection methods
- **MCP settings**: Server URLs, resource TTLs, tool configurations
- **Performance settings**: Caching parameters, worker counts, profiling options

Example configurations for specific use cases are available in the `config/presets` directory.

## Core Components

### Conversation Context Manager

The Context Manager maintains the conversation history and provides windowing capabilities for analysis:

```python
from mcp_therapist.core.context_manager import ConversationContextManager
from mcp_therapist.models.conversation import MessageRole

# Create a context manager
context_manager = ConversationContextManager()

# Create a new conversation
conversation = context_manager.create_conversation()

# Add a message
context_manager.add_message(
    conversation.id, 
    MessageRole.USER, 
    "Can you help me with my project?"
)

# Get current conversation window (last N messages)
window = context_manager.get_current_window(conversation.id, window_size=5)
```

### Conversation Model

The Conversation model defines the structure of conversations and messages:

```python
from mcp_therapist.models.conversation import Conversation, Message, MessageRole

# Create a conversation with messages
conversation = Conversation(
    id="conv_123",
    messages=[
        Message(
            id="msg_1",
            role=MessageRole.USER,
            content="Hello, how can you help me today?",
            timestamp=1637012400
        ),
        Message(
            id="msg_2",
            role=MessageRole.ASSISTANT,
            content="I'm here to answer your questions!",
            timestamp=1637012410
        )
    ],
    metadata={"session_id": "session_abc123"}
)
```

## Detectors

### Available Detectors

MCP Therapist includes multiple rut detectors:

1. **RepetitionDetector**: Identifies when the LLM repeats phrases or content
2. **StagnationDetector**: Detects when conversation isn't progressing or shows low diversity
3. **RefusalDetector**: Recognizes when the LLM refuses to fulfill reasonable requests
4. **ContradictionDetector**: Spots logical inconsistencies in the LLM's responses
5. **TopicFixationDetector**: Identifies when conversation is stuck on a narrow topic
6. **NegativityDetector**: Detects persistent negative sentiment in the conversation

### Detector Pipeline

The detector pipeline coordinates multiple detectors:

```python
from mcp_therapist.core.detectors.pipeline import DetectorPipeline
from mcp_therapist.core.detectors.registry import get_detector

# Create detector pipeline
pipeline = DetectorPipeline()

# Add detectors
pipeline.add_detector(get_detector("repetition"))
pipeline.add_detector(get_detector("stagnation"))
pipeline.add_detector(get_detector("refusal"))

# Run detection on a conversation
results = pipeline.detect(conversation)

# Process results
for detector_id, result in results.items():
    if result.detected:
        print(f"Detected {detector_id} with confidence {result.confidence}")
        print(f"Evidence: {result.metadata.get('evidence', '')}")
```

### Creating Custom Detectors

Custom detectors can be created by extending the base `RutDetector` class:

```python
from mcp_therapist.core.detectors.base import RutDetector, DetectionResult

class CustomDetector(RutDetector):
    """Custom detector to identify a specific pattern."""
    
    def __init__(self, threshold=0.7):
        """Initialize the detector."""
        super().__init__(id="custom_detector")
        self.threshold = threshold
    
    def detect(self, conversation):
        """Run detection on a conversation."""
        # Implement detection logic here
        confidence = self._calculate_confidence(conversation)
        detected = confidence >= self.threshold
        
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            metadata={
                "evidence": "Found evidence of pattern",
                "analysis": "Additional analysis details"
            }
        )
```

## Intervention System

### Intervention Manager

The Intervention Manager coordinates the complete intervention process:

```python
from mcp_therapist.core.interventions.manager import InterventionManager
from mcp_therapist.core.detectors.pipeline import DetectorPipeline

# Create components
detector_pipeline = DetectorPipeline()
# Add detectors...

# Create intervention manager
intervention_manager = InterventionManager(
    detector_pipeline=detector_pipeline
)

# Analyze conversation and intervene if needed
intervention_plan = intervention_manager.analyze_and_intervene(conversation)

# Check if intervention was created
if intervention_plan:
    print(f"Created intervention with strategy: {intervention_plan.strategy}")
    print(f"Prompt: {intervention_plan.prompt}")
```

### Intervention Strategies

The system includes multiple intervention strategies:

1. **Reframing**: Presents alternative perspectives on the current topic
2. **Metacognitive Prompting**: Encourages the LLM to reflect on its reasoning
3. **Topic Broadening**: Expands the conversation beyond the current narrow focus
4. **Constraint Clarification**: Helps clarify why the LLM is refusing certain requests
5. **Direct Guidance**: Provides specific direction to overcome a detected issue
6. **Self-reflection**: Prompts the LLM to evaluate its recent responses

### Injection Methods

Interventions can be delivered in several ways:

1. **DIRECT**: Adds the intervention as a separate message from a "Therapist"
2. **PREPEND**: Prepends the intervention to the user's next message
3. **INLINE**: Inserts the intervention in-line with the user's next message
4. **SELF_REFLECTION**: Adds a system message encouraging LLM self-reflection
5. **METADATA_ONLY**: Adds intervention information to message metadata only

```python
from mcp_therapist.core.interventions.injector import InterventionInjector, InjectionMethod

# Create injector
injector = InterventionInjector()

# Inject using different methods
injector.inject_intervention(
    conversation=conversation,
    intervention_text="Let's explore this from a different angle.",
    method=InjectionMethod.DIRECT
)
```

### Evaluation

The Evaluator measures intervention effectiveness:

```python
from mcp_therapist.core.interventions.evaluator import InterventionEvaluator

# Create evaluator
evaluator = InterventionEvaluator()

# Evaluate an intervention
success = evaluator.evaluate_intervention(
    conversation=conversation,
    intervention_id="intervention_123",
    window_size=3  # Check 3 messages after intervention
)

if success:
    print("Intervention was successful!")
else:
    print("Intervention did not resolve the issue.")
```

## MCP Integration

### MCP Client

The MCP Client handles communication with MCP servers:

```python
from mcp_therapist.mcp.client import MCPClient

# Create client
client = MCPClient(server_url="http://localhost:8000")

# Connect to server
await client.connect()

# Register a resource
resource_id = await client.register_resource({
    "type": "conversation",
    "data": conversation.to_dict()
})

# Close connection
await client.close()
```

### Resource Providers

Resource providers share data through the MCP:

```python
from mcp_therapist.mcp.resources import ConversationResourceProvider

# Create provider
provider = ConversationResourceProvider(
    get_conversation_func=my_get_conversation_function
)

# Register a conversation resource
resource_id = await provider.provide_conversation_resource(
    conversation_id="conv_123",
    window_size=10
)
```

### MCP Tools

MCP tools expose functionality to language models:

```python
from mcp_therapist.mcp.tools import InterventionToolProvider

# Create tool provider
tool_provider = InterventionToolProvider(
    intervention_manager=intervention_manager
)

# Register tools with an MCP client
await tool_provider.register_tools(mcp_client)
```

### Prompt Templates

The prompt system provides standardized templates:

```python
from mcp_therapist.mcp.prompts import MCPPromptManager

# Create prompt manager
prompt_manager = MCPPromptManager()

# Render a prompt template
prompt = await prompt_manager.render_prompt(
    prompt_id="metacognitive_intervention",
    parameters={
        "topic": "problem solving",
        "current_approach": "direct solution"
    }
)
```

### Factory Functions

The MCP factory simplifies system creation:

```python
from mcp_therapist.mcp.factory import create_mcp_system, shutdown_mcp_system

# Create complete MCP system
mcp_system = await create_mcp_system(
    intervention_manager=intervention_manager,
    get_conversation_func=get_conversation_function,
    server_url="http://localhost:8000"
)

# Use components
resource_providers = mcp_system["resource_providers"]
tool_providers = mcp_system["tool_providers"]
prompt_manager = mcp_system["prompt_manager"]
sampling_manager = mcp_system["sampling_manager"]

# Shut down system
await shutdown_mcp_system(mcp_system)
```

## Performance Optimization

### Profiling

The profiling system helps identify performance bottlenecks:

```python
from mcp_therapist.utils.profiling import Profiler

# Get global profiler
profiler = Profiler.get_instance()

# Enable profiling
profiler.enable()

# Profile a function
@profiler.profile_function
def expensive_function():
    # Function code...
    pass

# Profile a code section
with profiler.profile_section("my_operation"):
    # Code to profile...
    pass

# Get profiling statistics
stats = profiler.get_stats()
print(f"Total time: {stats['total_time_ms']} ms")

# Save detailed profile
profiler.save_profile("profile_output.prof")
```

### Caching

The caching system reduces redundant computations:

```python
from mcp_therapist.utils.caching import memoize, get_embedding_cache

# Use function memoization
@memoize(ttl=3600)  # Cache for 1 hour
def compute_embedding(text):
    # Expensive operation...
    return result

# Use global caches
embedding_cache = get_embedding_cache()
embedding_cache.set("key1", embedding_value)
cached_value = embedding_cache.get("key1")
```

### Parallel Processing

The concurrency utilities enable parallel execution:

```python
from mcp_therapist.utils.concurrency import task_manager, run_in_thread

# Run a function in a thread
@run_in_thread
def background_task():
    # Long-running operation...
    return result

# Start the task
future = background_task()

# Get the result when needed
result = future.result()

# Process items in batch
results = task_manager.run_in_batch(
    process_item,  # Function to apply to each item
    items,         # List of items to process
    max_batch_size=10
)
```

### Threshold Optimization

The threshold optimizer tunes detection parameters:

```python
from mcp_therapist.utils.threshold_optimizer import threshold_registry

# Get optimizer for a detector
optimizer = threshold_registry.get_optimizer(
    name="repetition_detector",
    initial_threshold=0.7
)

# Record detection results
optimizer.record_result(
    predicted=True,   # Detector predicted a rut
    actual=False,     # But it was a false positive
    confidence=0.75   # Detection confidence
)

# Get current performance metrics
metrics = optimizer.get_performance_metrics()
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")

# Optimize threshold based on collected data
result = optimizer.optimize_threshold(predictions)
print(f"New threshold: {result.threshold}")
```

## API Reference

Please see the [API.md](API.md) file for detailed API documentation.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mcp_therapist

# Run specific test module
pytest mcp_therapist/tests/detectors/test_repetition_detector.py
```

### Writing Tests

Use the template in `mcp_therapist/tests/template_test.py` as a starting point for new test files.

```python
import unittest
from unittest.mock import MagicMock, patch

class TestMyComponent(unittest.TestCase):
    def setUp(self):
        # Set up test fixtures
        self.mock_dependency = MagicMock()
        self.component = MyComponent(dependency=self.mock_dependency)
    
    def test_my_method(self):
        # Arrange
        expected_result = True
        
        # Act
        actual_result = self.component.my_method()
        
        # Assert
        self.assertEqual(expected_result, actual_result)
```

## Deployment

### Local Deployment

For local development and testing:

```bash
# Clone and install
git clone https://github.com/yourusername/mcp_therapist.git
cd mcp_therapist
pip install -r requirements.txt

# Run example script
python examples/basic_conversation.py
```

### Library Integration

To use as a library in another project:

```bash
# Install from git
pip install git+https://github.com/yourusername/mcp_therapist.git

# Import and use
from mcp_therapist.core.interventions.manager import InterventionManager
```

### Server-side Deployment

For integration in server environments:

1. Install in server environment:
   ```bash
   pip install git+https://github.com/yourusername/mcp_therapist.git
   ```

2. Set up environment variables in server environment

3. Import and integrate with your application

### Docker Deployment

```bash
# Build Docker image
docker build -t mcp_therapist .

# Run container
docker run -p 8080:8080 --env-file .env mcp_therapist
```

## Troubleshooting

### Common Issues

1. **ImportError or ModuleNotFoundError**:
   - Ensure you've installed all dependencies with `pip install -r requirements.txt`
   - Check your Python path includes the project directory

2. **API Key Errors**:
   - Verify you've set up `.env` file with correct API keys
   - Check for typos in key names or values

3. **MCP Connection Issues**:
   - Confirm MCP server is running and accessible
   - Check URL and port settings
   - Verify API key if server requires authentication

4. **Performance Problems**:
   - Enable profiling to identify bottlenecks
   - Adjust batch sizes in settings
   - Consider enabling caching mechanisms

### Logging

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set in environment:

```
LOG_LEVEL=DEBUG
```

### Getting Help

- Check the [FAQ](#faq) section below
- Search existing issues on GitHub
- Open a new issue with detailed information about your problem

## FAQ

**Q: Does MCP Therapist require API keys?**

A: Basic functionality works without API keys, but external LLM integrations (like OpenAI or Anthropic) require the corresponding API keys.

**Q: How does the system detect conversation ruts?**

A: The system uses multiple detection methods including pattern matching, semantic similarity, topic analysis, and sentiment tracking to identify various types of conversational ruts.

**Q: Can I add my own detector or intervention strategy?**

A: Yes, the system is designed to be extensible. You can create custom detectors by extending the `RutDetector` class and register custom strategies in the `InterventionStrategist`.

**Q: Is MCP integration required?**

A: No, MCP integration is optional. The core functionality works without MCP, but integration enables additional capabilities like resource sharing and external tool access.

**Q: How can I optimize performance for large conversations?**

A: Enable parallel processing, use caching, and consider adjusting detection thresholds. The `threshold_optimizer` can help fine-tune parameters for your specific use case.

**Q: Can the system work with any LLM?**

A: Yes, the system is designed to be model-agnostic. It works by analyzing conversation patterns rather than depending on specific model behaviors.

**Q: How can I contribute to this project?**

A: See the [Contributing](CONTRIBUTING.md) guide for information on how to contribute code, documentation, or ideas to MCP Therapist. 