"""Tests for langchain_agent.src.nodes.agent module."""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_agent.src.nodes.agent import (
    has_pending_tool_results,
    analyze_tool_results
)


class TestHasPendingToolResults:
    """Tests for the has_pending_tool_results function."""

    def test_empty_messages(self):
        """Empty message list should return False."""
        has_pending, tool_msgs = has_pending_tool_results([])
        assert has_pending is False
        assert tool_msgs == []

    def test_no_tool_messages(self):
        """No tool messages should return False."""
        messages = [
            HumanMessage(content="Turn on the light"),
            AIMessage(content="OK, turning on the light")
        ]
        has_pending, tool_msgs = has_pending_tool_results(messages)
        assert has_pending is False
        assert tool_msgs == []

    def test_tool_message_after_ai_with_tools(self):
        """Tool message after AI with tool_calls should return True."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "HassTurnOn", "args": {}}]

        messages = [
            HumanMessage(content="Turn on the light"),
            ai_msg,
            ToolMessage(content="Device turned on", tool_call_id="123")
        ]
        has_pending, tool_msgs = has_pending_tool_results(messages)
        assert has_pending is True
        assert len(tool_msgs) == 1

    def test_multiple_tool_messages(self):
        """Multiple tool messages should all be returned."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": "HassTurnOn", "args": {}},
            {"name": "HassTurnOff", "args": {}}
        ]

        messages = [
            HumanMessage(content="Turn on light and turn off fan"),
            ai_msg,
            ToolMessage(content="Light on", tool_call_id="1"),
            ToolMessage(content="Fan off", tool_call_id="2")
        ]
        has_pending, tool_msgs = has_pending_tool_results(messages)
        assert has_pending is True
        assert len(tool_msgs) == 2

    def test_ai_without_tool_calls(self):
        """AI message without tool_calls should return False."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            ToolMessage(content="Some content", tool_call_id="123")
        ]
        has_pending, tool_msgs = has_pending_tool_results(messages)
        assert has_pending is False

    def test_human_message_breaks_chain(self):
        """Human message should break tool message chain."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "HassTurnOn", "args": {}}]

        messages = [
            ai_msg,
            ToolMessage(content="Done", tool_call_id="1"),
            HumanMessage(content="Now do something else"),
        ]
        has_pending, tool_msgs = has_pending_tool_results(messages)
        assert has_pending is False


class TestAnalyzeToolResults:
    """Tests for the analyze_tool_results function."""

    def test_empty_tool_messages(self):
        """Empty list should return zeros."""
        result = analyze_tool_results([])
        assert result["total"] == 0
        assert result["successful"] == 0
        assert result["failed"] == 0
        assert result["details"] == []

    def test_successful_ha_action(self):
        """HA action_done should be marked as success."""
        tool_msg = ToolMessage(
            content="{'response_type': 'action_done'}",
            tool_call_id="123",
            name="HassTurnOn"
        )
        result = analyze_tool_results([tool_msg])
        assert result["successful"] == 1
        assert result["failed"] == 0
        assert result["details"][0]["status"] == "success"

    def test_failed_ha_action(self):
        """HA error response should be marked as failed."""
        tool_msg = ToolMessage(
            content="{'response_type': 'error', 'message': 'Device not found'}",
            tool_call_id="123",
            name="HassTurnOn"
        )
        result = analyze_tool_results([tool_msg])
        assert result["successful"] == 0
        assert result["failed"] == 1
        assert result["details"][0]["status"] == "failed"

    def test_general_error_keywords(self):
        """General error keywords should mark as failed."""
        tool_msg = ToolMessage(
            content="Unable to connect to device",
            tool_call_id="123",
            name="HassTurnOn"
        )
        result = analyze_tool_results([tool_msg])
        assert result["failed"] == 1

    def test_mixed_results(self):
        """Mixed success and failure should be counted correctly."""
        tool_msgs = [
            ToolMessage(
                content="{'response_type': 'action_done'}",
                tool_call_id="1",
                name="HassTurnOn"
            ),
            ToolMessage(
                content="Error: Device not found",
                tool_call_id="2",
                name="HassTurnOff"
            )
        ]
        result = analyze_tool_results(tool_msgs)
        assert result["total"] == 2
        assert result["successful"] == 1
        assert result["failed"] == 1

    def test_success_array_indicator(self):
        """Success array in response should indicate success."""
        tool_msg = ToolMessage(
            content='{"success": ["light.bedroom"], "failed": []}',
            tool_call_id="123",
            name="HassTurnOn"
        )
        result = analyze_tool_results([tool_msg])
        assert result["successful"] == 1

    def test_tool_name_captured(self):
        """Tool name should be captured in details."""
        tool_msg = ToolMessage(
            content="Done",
            tool_call_id="123",
            name="HassListAddItem"
        )
        result = analyze_tool_results([tool_msg])
        assert result["details"][0]["tool"] == "HassListAddItem"

    def test_content_preserved_in_details(self):
        """Original content should be in details."""
        tool_msg = ToolMessage(
            content="Item added to list",
            tool_call_id="123",
            name="HassListAddItem"
        )
        result = analyze_tool_results([tool_msg])
        assert result["details"][0]["content"] == "Item added to list"

    def test_empty_content(self):
        """Empty content should not raise."""
        tool_msg = ToolMessage(
            content="",
            tool_call_id="123",
            name="SomeTool"
        )
        result = analyze_tool_results([tool_msg])
        # Empty content without error keywords = success
        assert result["successful"] == 1

    def test_none_content(self):
        """None content should be handled."""
        tool_msg = ToolMessage(
            content=None,
            tool_call_id="123",
            name="SomeTool"
        )
        result = analyze_tool_results([tool_msg])
        assert result["total"] == 1
