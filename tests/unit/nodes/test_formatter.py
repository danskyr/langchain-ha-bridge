"""Tests for langchain_agent.src.nodes.formatter module."""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_agent.src.nodes.formatter import formatter_node


class TestFormatterNode:
    """Tests for the formatter_node function."""

    def test_formats_ai_message_content(self):
        """Should extract content from AIMessage."""
        state = {
            "messages": [
                HumanMessage(content="Turn on the light"),
                AIMessage(content="I've turned on the living room light.")
            ],
            "query": "Turn on the light",
            "route_types": ["iot"],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = formatter_node(state)

        assert "final_response" in result
        assert result["final_response"] == "I've turned on the living room light."

    def test_continues_conversation_for_question(self):
        """Should set continue_conversation=True for questions."""
        state = {
            "messages": [
                AIMessage(content="Which light would you like me to turn on?")
            ],
            "query": "Turn on the light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = formatter_node(state)

        assert result["continue_conversation"] is True

    def test_ends_conversation_for_statement(self):
        """Should set continue_conversation=False for statements."""
        state = {
            "messages": [
                AIMessage(content="I've turned on the bedroom light.")
            ],
            "query": "Turn on the light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = formatter_node(state)

        assert result["continue_conversation"] is False

    def test_continues_for_semicolon(self):
        """Should set continue_conversation=True for semicolon endings."""
        state = {
            "messages": [
                AIMessage(content="Please specify the room;")
            ],
            "query": "Turn on the light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = formatter_node(state)

        assert result["continue_conversation"] is True

    def test_handles_non_message_object(self):
        """Should convert non-message objects to string."""
        state = {
            "messages": ["Plain string message"],
            "query": "Test",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = formatter_node(state)

        assert result["final_response"] == "Plain string message"

    def test_handles_chinese_question_mark(self):
        """Should recognize Chinese question mark."""
        state = {
            "messages": [
                AIMessage(content="您想开哪个灯？")
            ],
            "query": "Turn on the light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = formatter_node(state)

        assert result["continue_conversation"] is True

    def test_strips_whitespace_before_checking(self):
        """Should strip whitespace before checking for question mark."""
        state = {
            "messages": [
                AIMessage(content="Which light?   ")
            ],
            "query": "Turn on the light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = formatter_node(state)

        assert result["continue_conversation"] is True
