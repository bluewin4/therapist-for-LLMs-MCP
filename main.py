#!/usr/bin/env python3
"""
Main entry point for the MCP Therapist system.
This script demonstrates the basic functionality of the system.
"""

import argparse
import sys
from typing import Dict, List, Optional

from mcp_therapist import __version__
from mcp_therapist.core.context_manager import ConversationContextManager
from mcp_therapist.models.conversation import MessageRole
from mcp_therapist.utils.logging import logger, setup_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MCP Therapist - LLM interaction monitoring system")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-file", help="Log file path")
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment based on command-line arguments."""
    if args.debug:
        logger = setup_logger("mcp_therapist", level="DEBUG", file_path=args.log_file)
    elif args.log_file:
        logger = setup_logger("mcp_therapist", file_path=args.log_file)
    
    return logger


def demo():
    """Run a simple demonstration of the system's basic functionality."""
    # Create a conversation manager
    context_manager = ConversationContextManager()
    
    # Create a new conversation
    logger.info("Creating a new conversation...")
    conversation = context_manager.create_conversation()
    
    # Demo conversation
    logger.info(f"Created conversation with ID: {conversation.id}")
    
    # Add some messages to simulate a conversation
    messages = [
        (MessageRole.USER, "Hello, can you help me solve a programming problem?"),
        (MessageRole.ASSISTANT, "Of course! I'd be happy to help with your programming problem. What seems to be the issue?"),
        (MessageRole.USER, "I'm trying to write a Python function to calculate Fibonacci numbers efficiently."),
        (MessageRole.ASSISTANT, "That's a great problem to work on! There are several approaches to calculating Fibonacci numbers efficiently in Python. Let me walk you through a few options:"),
        (MessageRole.ASSISTANT, "1. **Dynamic Programming (Bottom-up)**: This is an iterative approach that avoids recursion overhead.\n2. **Memoization (Top-down)**: This uses recursion but caches previous results.\n3. **Matrix Exponentiation**: This can calculate Fibonacci numbers in O(log n) time."),
        (MessageRole.USER, "The iterative approach sounds simplest. How would I implement that?"),
        (MessageRole.ASSISTANT, "Here's how you can implement the iterative approach to calculate Fibonacci numbers:"),
    ]
    
    for role, content in messages:
        logger.info(f"Adding message: {role.value} - {content[:30]}...")
        context_manager.add_message(conversation.id, role, content)
    
    # Demonstrate accessing the conversation window
    window_size = 3
    logger.info(f"Getting conversation window (size={window_size})...")
    window = context_manager.get_current_window(conversation.id, size=window_size)
    
    logger.info(f"Last {len(window)} messages:")
    for i, message in enumerate(window):
        logger.info(f"  [{i+1}] {message.role.value}: {message.content[:50]}...")
    
    # Demonstrate serialization/deserialization
    logger.info("Converting conversation to dictionary...")
    conv_dict = context_manager.to_dict(conversation.id)
    
    logger.info(f"Dictionary representation has {len(conv_dict['messages'])} messages")
    
    logger.info("Restoring conversation from dictionary...")
    new_manager = ConversationContextManager()
    restored_conv = new_manager.from_dict(conv_dict)
    
    logger.info(f"Restored conversation has {len(restored_conv.messages)} messages")
    
    # Success message
    logger.info("Demo completed successfully!")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print version and exit if requested
    if args.version:
        print(f"MCP Therapist version {__version__}")
        sys.exit(0)
    
    # Set up the environment
    logger = setup_environment(args)
    
    # Run the demo
    logger.info("Starting MCP Therapist demo...")
    demo()
    logger.info("MCP Therapist demo completed")


if __name__ == "__main__":
    main() 