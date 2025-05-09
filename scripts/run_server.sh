#!/bin/bash
# Example script to run the LangChain server

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Set the models to use
export ROUTER_MODEL="gpt-3.5-turbo"
export QUERY_MODEL="gpt-4"

# Run the server
python -m src.langchain_ha_bridge

# Alternatively, you can run the server directly:
# python langchain_agent/langchain_ha_bridge/server.py