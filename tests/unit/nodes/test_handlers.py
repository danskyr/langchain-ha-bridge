"""Tests for langchain_agent.src.nodes.handlers module."""
import pytest
from langchain_agent.src.nodes.handlers import (
    iot_handler_node,
    general_handler_node
)


class TestIotHandlerNode:
    """Tests for the iot_handler_node function."""

    def test_returns_handler_response(self):
        """IOT handler should return response with handler name."""
        state = {
            "messages": [],
            "query": "turn on the light",
            "route_types": ["iot"],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = iot_handler_node(state)

        assert "handler_responses" in result
        assert len(result["handler_responses"]) == 1
        assert result["handler_responses"][0]["handler"] == "iot"

    def test_includes_confidence(self):
        """IOT handler should include confidence score."""
        state = {
            "messages": [],
            "query": "turn on the light",
            "route_types": ["iot"],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = iot_handler_node(state)

        assert result["handler_responses"][0]["confidence"] == 0.8


class TestGeneralHandlerNode:
    """Tests for the general_handler_node function."""

    def test_returns_handler_response(self):
        """General handler should return response with handler name."""
        state = {
            "messages": [],
            "query": "what's the weather",
            "route_types": ["general"],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = general_handler_node(state)

        assert "handler_responses" in result
        assert len(result["handler_responses"]) == 1
        assert result["handler_responses"][0]["handler"] == "general"

    def test_includes_confidence(self):
        """General handler should include confidence score."""
        state = {
            "messages": [],
            "query": "hello",
            "route_types": ["general"],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = general_handler_node(state)

        assert result["handler_responses"][0]["confidence"] == 0.7
