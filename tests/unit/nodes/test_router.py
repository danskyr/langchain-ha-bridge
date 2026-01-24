"""Tests for langchain_agent.src.nodes.router module."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from langchain_agent.src.nodes.router import router_node, route_to_handlers


class TestRouterNode:
    """Tests for the router_node function."""

    @patch('langchain_agent.src.nodes.router.classify_intent')
    def test_routes_iot_intent(self, mock_classify):
        """IOT intent should route to iot."""
        mock_classify.return_value = "iot"

        state = {
            "messages": [HumanMessage(content="turn on the light")],
            "query": "turn on the light",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = router_node(state)

        assert "route_types" in result
        assert "iot" in result["route_types"]

    @patch('langchain_agent.src.nodes.router.classify_intent')
    def test_routes_search_intent(self, mock_classify):
        """Search intent should route to search."""
        mock_classify.return_value = "search"

        state = {
            "messages": [HumanMessage(content="what's the weather")],
            "query": "what's the weather",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = router_node(state)

        assert "route_types" in result
        assert "search" in result["route_types"]

    @patch('langchain_agent.src.nodes.router.classify_intent')
    def test_routes_general_intent(self, mock_classify):
        """General intent should route to general."""
        mock_classify.return_value = "general"

        state = {
            "messages": [HumanMessage(content="hello")],
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

        result = router_node(state)

        assert "route_types" in result
        assert "general" in result["route_types"]

    @patch('langchain_agent.src.nodes.router.classify_intent')
    def test_defaults_to_general(self, mock_classify):
        """Unknown intent should default to general."""
        mock_classify.return_value = "unknown"

        state = {
            "messages": [HumanMessage(content="xyz123")],
            "query": "xyz123",
            "route_types": [],
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        result = router_node(state)

        assert "route_types" in result
        assert "general" in result["route_types"]


class TestRouteToHandlers:
    """Tests for the route_to_handlers function."""

    def test_iot_route_returns_iot_handler(self):
        """IOT route should return iot_handler."""
        state = {
            "route_types": ["iot"],
            "messages": [],
            "query": "",
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        handlers = route_to_handlers(state)

        assert "iot_handler" in handlers

    def test_general_route_returns_general_handler(self):
        """General route should return general_handler."""
        state = {
            "route_types": ["general"],
            "messages": [],
            "query": "",
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        handlers = route_to_handlers(state)

        assert "general_handler" in handlers

    def test_search_route_returns_general_handler(self):
        """Search route should return general_handler (has Tavily)."""
        state = {
            "route_types": ["search"],
            "messages": [],
            "query": "",
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        handlers = route_to_handlers(state)

        assert "general_handler" in handlers

    def test_empty_route_defaults_to_general_handler(self):
        """Empty route_types should default to general_handler."""
        state = {
            "route_types": [],
            "messages": [],
            "query": "",
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        handlers = route_to_handlers(state)

        assert "general_handler" in handlers

    def test_missing_route_types_defaults(self):
        """Missing route_types key should default to general."""
        state = {
            "messages": [],
            "query": "",
            "handler_responses": [],
            "final_response": None,
            "tools": [],
            "validation_attempts": 1,
            "continue_conversation": None,
            "preliminary_messages": [],
            "streaming_events": [],
        }

        handlers = route_to_handlers(state)

        assert "general_handler" in handlers
