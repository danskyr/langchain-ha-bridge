"""Pytest configuration and shared fixtures."""
import logging
import os
import sys
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Set environment variables before any imports that might need them
os.environ.setdefault("LOG_EXECUTION_PATH", "true")
os.environ.setdefault("CAPTURE_MOCK_DATA", "false")

# Add custom_components to path for import (not a poetry package)
project_root = Path(__file__).parent.parent
custom_components_dir = project_root / "custom_components"
sys.path.insert(0, str(custom_components_dir))


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def sample_router_state():
    """Create a sample RouterState for testing."""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content="Turn on the living room light")],
        "query": "Turn on the living room light",
        "route_types": [],
        "handler_responses": [],
        "final_response": None,
        "tools": [],
        "validation_attempts": 1,
        "continue_conversation": None,
        "preliminary_messages": [],
        "streaming_events": [],
    }


@pytest.fixture
def sample_tool_schema():
    """Create a sample tool schema for validation testing."""
    return {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "description": "Turn on a device",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {"type": "string", "description": "The area name"},
                    "domain": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["light", "switch", "fan"]}
                    },
                    "device_class": {
                        "type": "string",
                        "enum": ["switch", "outlet", "plug"]
                    }
                },
                "required": []
            }
        }
    }


@pytest.fixture
def sample_ha_tools():
    """Create sample Home Assistant tools for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "HassTurnOn",
                "description": "Turn on a device",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string"},
                        "name": {"type": "string"},
                        "domain": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "HassTurnOff",
                "description": "Turn off a device",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string"},
                        "name": {"type": "string"},
                        "domain": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": []
                }
            }
        }
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for file-based tests."""
    return tmp_path
