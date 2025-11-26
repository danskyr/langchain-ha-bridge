import logging
from typing import Dict, Any
from ..state import RouterState

logger = logging.getLogger('langchain_agent.nodes.announcement')


def announcement_node(state: RouterState) -> Dict[str, Any]:
    """
    Generate preliminary announcement messages based on route types and query.

    This node runs in parallel with handlers to provide immediate user feedback
    while processing continues in the background.
    """
    route_types = state.get("route_types", [])
    query = state.get("query", "")
    validation_attempts = state.get("validation_attempts", 1)

    logger.info(f"[announcement] Routes: {route_types}, Validation attempt: {validation_attempts}")

    preliminary_messages = []

    # Generate context-aware preliminary messages
    if validation_attempts > 1:
        # We're retrying after validation failure
        preliminary_messages.append("Let me correct that and try again...")

    elif "search" in route_types:
        # Web search will take time
        if "weather" in query.lower():
            preliminary_messages.append("Let me check the weather for you...")
        elif "news" in query.lower():
            preliminary_messages.append("Let me search for the latest news...")
        else:
            preliminary_messages.append("Let me search for that information...")

    elif "iot" in route_types and len(route_types) > 1:
        # Complex IOT operation
        preliminary_messages.append("Let me help you with those devices...")

    elif any(keyword in query.lower() for keyword in ["analyze", "summarize", "explain", "compare"]):
        # Complex analysis task
        preliminary_messages.append("Let me analyze that for you...")

    if preliminary_messages:
        logger.info(f"[announcement] Emitting: {preliminary_messages[0]}")

        # Also create a streaming event for immediate delivery
        streaming_event = {
            "type": "preliminary",
            "content": preliminary_messages[0]
        }

        return {
            "preliminary_messages": preliminary_messages,
            "streaming_events": [streaming_event]
        }

    logger.info("[announcement] No announcement needed for this query")
    return {}