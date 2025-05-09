#!/usr/bin/env python3
"""Test script for LangChain Router Agent."""
import asyncio
import os
import sys
import yaml
from typing import Dict, Any

# Add the parent directory to the path so we can import the custom component
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from custom_components.langchain_agent.conversation import LangChainRouterAgent

class MockHomeAssistant:
    """Mock Home Assistant class for testing."""
    
    def __init__(self):
        """Initialize the mock."""
        self.data = {}
        
    async def async_add_executor_job(self, func, *args, **kwargs):
        """Mock async_add_executor_job."""
        return func(*args, **kwargs)

async def test_router_agent():
    """Test the LangChain Router Agent."""
    # Get OpenAI API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Create mock Home Assistant
    hass = MockHomeAssistant()
    
    # Create config
    config = {
        "openai_api_key": openai_api_key,
        "router_model": "gpt-3.5-turbo",
        "query_model": "gpt-3.5-turbo",  # Using 3.5 for testing to save costs
    }
    
    # Create agent
    agent = LangChainRouterAgent(hass, config)
    
    # Test device control queries
    device_queries = [
        "Turn on the living room lights",
        "Set the thermostat to 72 degrees",
        "Turn off all the lights in the house",
    ]
    
    # Test general queries
    general_queries = [
        "What is home automation?",
        "How does a smart thermostat work?",
        "Tell me a joke about smart homes",
    ]
    
    print("\n=== Testing Device Control Queries ===")
    for query in device_queries:
        print(f"\nQuery: {query}")
        result = await agent.async_process(query)
        print(f"Response: {result.response}")
        
        # Try to parse the response as YAML to verify it's a valid service call
        try:
            service_call = yaml.safe_load(result.response)
            print(f"Parsed service call: {service_call}")
        except Exception as e:
            print(f"Error parsing response as YAML: {e}")
    
    print("\n=== Testing General Queries ===")
    for query in general_queries:
        print(f"\nQuery: {query}")
        result = await agent.async_process(query)
        print(f"Response: {result.response}")

if __name__ == "__main__":
    asyncio.run(test_router_agent())