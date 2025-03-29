#!/usr/bin/env python
"""
MCP Integration Example

This script demonstrates how to use the MCP integration components
of the MCP Therapist system.
"""

import asyncio
import uuid
import json
from datetime import datetime

from mcp_therapist.models.conversation import (
    Conversation, 
    Message, 
    RutType, 
    InterventionStrategy
)
from mcp_therapist.core.interventions.manager import InterventionManager
from mcp_therapist.core.interventions.injector import InterventionInjector
from mcp_therapist.core.interventions.evaluator import InterventionEvaluator
from mcp_therapist.mcp.factory import create_mcp_system, shutdown_mcp_system
from mcp_therapist.mcp.sampling import SamplingContext, SamplingParameters


# Sample conversation for demonstration
def create_sample_conversation():
    conversation = Conversation(
        id=str(uuid.uuid4()),
        messages=[
            Message(
                id=str(uuid.uuid4()),
                role="user",
                content="I've been feeling really stressed lately with work.",
                timestamp=datetime.now().isoformat()
            ),
            Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content="I'm sorry to hear that you're feeling stressed with work. What aspects of your work are causing the most stress?",
                timestamp=datetime.now().isoformat()
            ),
            Message(
                id=str(uuid.uuid4()),
                role="user",
                content="It's just so much pressure all the time. I can't handle it.",
                timestamp=datetime.now().isoformat()
            ),
            Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content="That sounds really challenging. Constant pressure can be overwhelming. Are there specific situations at work that feel particularly stressful?",
                timestamp=datetime.now().isoformat()
            ),
            Message(
                id=str(uuid.uuid4()),
                role="user",
                content="Just deadlines and expectations. It's too much pressure. I'm constantly anxious about it.",
                timestamp=datetime.now().isoformat()
            )
        ],
        metadata={}
    )
    return conversation


# Retrieve a conversation by ID
async def get_conversation(conversation_id):
    # This is a placeholder - in a real system, this would access a database
    if hasattr(get_conversation, "sample_conversation") and \
       get_conversation.sample_conversation.id == conversation_id:
        return get_conversation.sample_conversation
    return None


# Example of using MCP resources
async def demonstrate_resources(mcp_system, conversation):
    print("\n=== MCP Resources Demonstration ===")
    
    # Get the resource providers
    conversation_provider = mcp_system["resource_providers"]["conversation"]
    
    # Provide the conversation as a resource
    resource_id = await conversation_provider.provide_conversation_resource(
        conversation=conversation,
        window_size=3  # Only include the last 3 messages
    )
    
    print(f"Provided conversation as resource: {resource_id}")
    
    # Create a rut detection result
    detection_result = {
        "rut_type": RutType.STUCK_ON_EMOTION,
        "confidence": 0.85,
        "context": {
            "emotion": "anxiety",
            "indicators": ["pressure", "anxious", "can't handle it"],
            "pattern": "repeating emotional state without deeper exploration"
        }
    }
    
    # Provide the detection result as a resource
    detection_resource_id = await conversation_provider.provide_rut_detection_resource(
        conversation=conversation,
        detection_result=detection_result
    )
    
    print(f"Provided rut detection as resource: {detection_resource_id}")
    
    return [resource_id, detection_resource_id]


# Example of using MCP prompts
async def demonstrate_prompts(mcp_system):
    print("\n=== MCP Prompts Demonstration ===")
    
    # Get the prompt manager
    prompt_manager = mcp_system["prompt_manager"]
    
    # List available prompts
    prompts = await prompt_manager.list_prompts()
    print(f"Available prompts: {len(prompts)}")
    for prompt in prompts[:3]:  # Show just a few
        print(f"  - {prompt['id']}: {prompt['name']}")
    
    # Render a direct intervention prompt
    intervention_text = await prompt_manager.render_prompt(
        prompt_id="direct_intervention",
        parameters={
            "intervention_content": "I've noticed that we're focusing a lot on the feeling of pressure, which is important, but perhaps we could explore some specific strategies that might help manage these feelings."
        }
    )
    
    print("\nRendered direct intervention prompt:")
    print(intervention_text)
    
    # Render a self-reflection prompt
    reflection_text = await prompt_manager.render_prompt(
        prompt_id="self_reflection_intervention",
        parameters={
            "rut_type": "Stuck on emotion",
            "analysis": "The user is repeatedly expressing feelings of anxiety and pressure without moving toward problem-solving or deeper understanding.",
            "strategy_1": "Guide toward specific examples",
            "strategy_2": "Explore coping mechanisms",
            "strategy_3": "Validate emotions while gently shifting focus"
        }
    )
    
    print("\nRendered self-reflection prompt:")
    print(reflection_text)
    
    return intervention_text


# Example of using MCP sampling
async def demonstrate_sampling(mcp_system, resource_ids, prompt_text):
    print("\n=== MCP Sampling Demonstration ===")
    
    # Get the sampling manager
    sampling_manager = mcp_system["sampling_manager"]
    
    # Create a sampling context
    context = SamplingContext(
        system_prompt="You are a therapeutic assistant helping a user manage work stress. Use a compassionate, empathetic approach.",
        resource_ids=resource_ids,
        user_prompt="Based on our conversation so far and the pattern of emotional focus, how would you respond to help the user explore their stress more productively?"
    )
    
    # Create sampling parameters
    parameters = SamplingParameters(
        temperature=0.7,
        max_tokens=500,
        model="gpt-4"
    )
    
    # Sample text
    print("Sampling therapeutic response...")
    result = await sampling_manager.sample_text(context, parameters)
    
    print(f"\nSampled text ({result.finish_reason}, {result.usage.get('total_tokens', 0)} tokens):")
    print(result.text)
    
    # Demonstrate multi-step reasoning
    print("\nDemonstrating multi-step reasoning...")
    reasoning_result = await sampling_manager.multi_step_reasoning(
        question="How can I help the user explore the specific causes of their work stress and develop effective coping strategies?",
        steps=2,
        resource_ids=resource_ids
    )
    
    print(f"\nMulti-step reasoning - Final answer:")
    print(reasoning_result["final_answer"])
    
    return result.text


# Main demonstration
async def main():
    print("=== MCP Integration Example ===")
    
    # Create a sample conversation
    conversation = create_sample_conversation()
    
    # Store it for the get_conversation function
    get_conversation.sample_conversation = conversation
    
    # Create intervention components
    injector = InterventionInjector()
    evaluator = InterventionEvaluator()
    intervention_manager = InterventionManager(injector=injector, evaluator=evaluator)
    
    # Create the MCP system
    print("Initializing MCP system...")
    mcp_system = await create_mcp_system(
        intervention_manager=intervention_manager,
        get_conversation_func=get_conversation
    )
    
    if not mcp_system["enabled"]:
        print("MCP integration is disabled in settings. Exiting.")
        return
    
    try:
        # Demonstrate MCP resources
        resource_ids = await demonstrate_resources(mcp_system, conversation)
        
        # Demonstrate MCP prompts
        prompt_text = await demonstrate_prompts(mcp_system)
        
        # Demonstrate MCP sampling
        response = await demonstrate_sampling(mcp_system, resource_ids, prompt_text)
        
        # Create a new message with the response
        new_message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat()
        )
        
        # Add it to the conversation
        conversation.messages.append(new_message)
        
        print("\n=== Updated Conversation ===")
        for i, msg in enumerate(conversation.messages):
            role = "User:    " if msg.role == "user" else "Assistant:"
            print(f"{i+1}. {role} {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
    
    finally:
        # Shut down the MCP system
        print("\nShutting down MCP system...")
        await shutdown_mcp_system(mcp_system)
    
    print("\nMCP integration example completed.")


if __name__ == "__main__":
    asyncio.run(main()) 