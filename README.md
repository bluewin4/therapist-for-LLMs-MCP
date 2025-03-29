# MCP Therapist

A system that monitors LLM interactions, detects conversational "ruts," and provides therapeutic interventions to guide the LLM towards more beneficial states.

## Overview

MCP Therapist runs alongside a primary LLM interaction, analyzing the conversation history and injecting "Infoblessing" prompts when it detects the LLM is stuck in an unhelpful state. The system is designed to be modular and extensible, with support for integration with the Model Context Protocol (MCP).

## Project Structure

```
mcp_therapist/
├── core/               # Core system components
│   ├── context_manager.py    # Manages conversation contexts
│   ├── detectors/            # Rut detection modules
│   └── interventions/        # Intervention components
├── models/             # Data models
│   └── conversation.py       # Conversation-related models
├── mcp/                # MCP integration components
│   ├── client.py             # MCP client implementation
│   ├── resources.py          # MCP resource providers
│   ├── tools.py              # MCP tool implementations
│   ├── prompts.py            # MCP prompt templates
│   └── sampling.py           # MCP sampling capabilities
├── utils/              # Utility functions
├── config/             # Configuration files
└── tests/              # Test cases
```

## Features

- Monitors ongoing LLM conversations 
- Detects various conversation "ruts" (repetition, stagnation, refusal, etc.)
- Selects appropriate intervention strategies
- Generates and injects therapeutic prompts
- Evaluates intervention effectiveness
- MCP integration for enhanced context sharing and tool access

### MCP Integration Features

The MCP Therapist system includes comprehensive integration with the Model Context Protocol (MCP), providing:

- **Resource Sharing**: Share conversation history, rut detections, interventions, and user profiles with language models.
- **Tool Exposure**: Provide tools for conversation analysis, intervention creation, and evaluation.
- **Prompt Templates**: Standardized therapeutic prompts for consistent interactions.
- **Recursive LLM Sampling**: Support for multi-step reasoning and enhanced therapeutic reflections.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp_therapist.git
cd mcp_therapist

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from mcp_therapist.core.context_manager import ConversationContextManager
from mcp_therapist.models.conversation import MessageRole

# Create a conversation manager
context_manager = ConversationContextManager()

# Create a new conversation
conversation = context_manager.create_conversation()

# Add messages to the conversation
context_manager.add_message(conversation.id, MessageRole.USER, "Hello, can you help me?")
context_manager.add_message(conversation.id, MessageRole.ASSISTANT, "Yes, I'd be happy to help. What do you need?")

# Get the current conversation window
window = context_manager.get_current_window(conversation.id)
```

### MCP Integration Usage

```python
import asyncio
from mcp_therapist.models.conversation import Conversation, Message
from mcp_therapist.core.interventions.manager import InterventionManager
from mcp_therapist.mcp.factory import create_mcp_system, shutdown_mcp_system

async def main():
    # Create intervention manager
    intervention_manager = InterventionManager()
    
    # Initialize MCP system
    mcp_system = await create_mcp_system(
        intervention_manager=intervention_manager,
        get_conversation_func=my_get_conversation_function
    )
    
    try:
        # Get resource providers
        conversation_provider = mcp_system["resource_providers"]["conversation"]
        
        # Share a conversation as an MCP resource
        resource_id = await conversation_provider.provide_conversation_resource(
            conversation=my_conversation,
            window_size=10  # Last 10 messages
        )
        
        # Use the prompt manager
        prompt_manager = mcp_system["prompt_manager"]
        intervention_text = await prompt_manager.render_prompt(
            prompt_id="direct_intervention",
            parameters={"intervention_content": "Let's explore this differently..."}
        )
        
        # Use LLM sampling
        sampling_manager = mcp_system["sampling_manager"]
        reflection = await sampling_manager.create_therapy_reflection(
            conversation_resource_id=resource_id,
            focus_area="emotional patterns"
        )
        
    finally:
        # Shut down MCP system
        await shutdown_mcp_system(mcp_system)

if __name__ == "__main__":
    asyncio.run(main())
```

Check the `examples/` directory for more usage examples.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/my-feature`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Model Context Protocol (MCP) specification
- Conceptual framework for understanding LLM behavior
