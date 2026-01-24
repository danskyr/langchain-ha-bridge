"""Tests for langchain_agent.src.nodes.validation module."""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_agent.src.nodes.validation import (
    separate_tool_calls,
    validation_decision,
    LOCAL_TOOLS
)


class TestSeparateToolCalls:
    """Tests for the separate_tool_calls function."""

    def test_empty_list(self):
        """Empty list should return empty tuples."""
        ha_tools, local_tools = separate_tool_calls([])
        assert ha_tools == []
        assert local_tools == []

    def test_ha_tools_only(self):
        """HA tools should be separated correctly."""
        tool_calls = [
            {"name": "HassTurnOn", "args": {}},
            {"name": "HassTurnOff", "args": {}},
            {"name": "HassListAddItem", "args": {}}
        ]
        ha_tools, local_tools = separate_tool_calls(tool_calls)
        assert len(ha_tools) == 3
        assert len(local_tools) == 0

    def test_local_tools_only(self):
        """Local tools should be separated correctly."""
        tool_calls = [
            {"name": "tavily_web_search", "args": {}}
        ]
        ha_tools, local_tools = separate_tool_calls(tool_calls)
        assert len(ha_tools) == 0
        assert len(local_tools) == 1

    def test_mixed_tools(self):
        """Mixed tools should be separated correctly."""
        tool_calls = [
            {"name": "HassTurnOn", "args": {}},
            {"name": "tavily_web_search", "args": {}},
            {"name": "HassTurnOff", "args": {}}
        ]
        ha_tools, local_tools = separate_tool_calls(tool_calls)
        assert len(ha_tools) == 2
        assert len(local_tools) == 1
        assert ha_tools[0]["name"] == "HassTurnOn"
        assert local_tools[0]["name"] == "tavily_web_search"

    def test_unknown_tools_go_to_ha(self):
        """Unknown tools should be assumed as HA tools."""
        tool_calls = [
            {"name": "SomeNewTool", "args": {}}
        ]
        ha_tools, local_tools = separate_tool_calls(tool_calls)
        assert len(ha_tools) == 1
        assert len(local_tools) == 0


class TestLocalToolsConstant:
    """Tests for the LOCAL_TOOLS constant."""

    def test_tavily_is_local(self):
        """Tavily web search should be in LOCAL_TOOLS."""
        assert "tavily_web_search" in LOCAL_TOOLS


class TestValidationDecision:
    """Tests for the validation_decision function."""

    def test_final_response_goes_to_formatter(self):
        """If final_response is set, should go to formatter."""
        state = {
            "messages": [AIMessage(content="Error message")],
            "query": "test",
            "route_types": [],
            "handler_responses": [],
            "final_response": "Some error occurred",
            "tools": [],
            "validation_attempts": 4,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = validation_decision(state)
        assert result == "formatter"

    def test_no_tool_calls_goes_to_formatter(self):
        """No tool calls should go to formatter."""
        state = {
            "messages": [AIMessage(content="Hello!")],
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

        result = validation_decision(state)
        assert result == "formatter"

    def test_ha_tools_returns_ha_tools(self):
        """HA tool calls should return ha_tools."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "HassTurnOn", "args": {}}]

        state = {
            "messages": [ai_msg],
            "query": "turn on light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = validation_decision(state)
        assert result == "ha_tools"

    def test_local_tools_returns_local_tools(self):
        """Local tool calls should return local_tools."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "tavily_web_search", "args": {}}]

        state = {
            "messages": [ai_msg],
            "query": "search for weather",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = validation_decision(state)
        assert result == "local_tools"

    def test_mixed_tools_prioritizes_ha(self):
        """Mixed HA and local tools should prioritize HA."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": "HassTurnOn", "args": {}},
            {"name": "tavily_web_search", "args": {}}
        ]

        state = {
            "messages": [ai_msg],
            "query": "turn on light and search",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = validation_decision(state)
        assert result == "ha_tools"

    def test_retry_on_validation_error(self):
        """Validation error (attempt > 1, no real tool calls) should retry."""
        # AIMessage with content but no tool_calls indicates validation error
        ai_msg = AIMessage(content="Validation failed: invalid parameter")

        state = {
            "messages": [ai_msg],
            "query": "turn on light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 2,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = validation_decision(state)
        assert result == "retry"

    def test_empty_tool_calls_goes_to_formatter(self):
        """Empty tool_calls list should go to formatter."""
        ai_msg = AIMessage(content="I can help with that")
        ai_msg.tool_calls = []

        state = {
            "messages": [ai_msg],
            "query": "hello",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = validation_decision(state)
        assert result == "formatter"
