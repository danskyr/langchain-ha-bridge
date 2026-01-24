"""Tests for langchain_agent.src.nodes.aggregator module."""
import pytest
from langchain_agent.src.nodes.aggregator import aggregator_node


class TestAggregatorNode:
    """Tests for the aggregator_node function."""

    def test_returns_empty_dict(self):
        """Aggregator should return empty dict (no state modifications)."""
        state = {
            "messages": [],
            "query": "test",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = aggregator_node(state)

        assert result == {}

    def test_handles_single_handler_response(self):
        """Should handle single handler response."""
        state = {
            "messages": [],
            "query": "test",
            "route_types": ["iot"],
            "handler_responses": [{"handler": "iot", "confidence": 0.8}],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = aggregator_node(state)

        assert result == {}

    def test_handles_multiple_handler_responses(self):
        """Should handle multiple handler responses."""
        state = {
            "messages": [],
            "query": "test",
            "route_types": ["iot", "general"],
            "handler_responses": [
                {"handler": "iot", "confidence": 0.8},
                {"handler": "general", "confidence": 0.7}
            ],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = aggregator_node(state)

        assert result == {}

    def test_handles_empty_handler_responses(self):
        """Should handle empty handler responses list."""
        state = {
            "messages": [],
            "query": "test",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = aggregator_node(state)

        assert result == {}

    def test_handles_missing_handler_responses_key(self):
        """Should handle state without handler_responses key."""
        state = {
            "messages": [],
            "query": "test",
            "route_types": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = aggregator_node(state)

        assert result == {}
