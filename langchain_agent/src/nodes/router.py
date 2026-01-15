import logging
from typing import Dict, Any, Sequence
from ..state import RouterState
from ..utils import preview_text
from ..semantic_router import classify_intent

logger = logging.getLogger('langchain_agent.nodes.router')


def router_node(state: RouterState) -> Dict[str, Any]:
    """Determine which handlers should process this query using semantic routing."""
    query = state["query"]
    logger.info(f"[router] Analyzing query: {preview_text(query, 100)}")

    intent = classify_intent(query)

    route_types = []
    if intent == "iot":
        route_types.append("iot")
    elif intent == "search":
        route_types.append("search")
    else:
        route_types.append("general")

    logger.info(f"[router] Routes selected: {route_types}")

    return {
        "route_types": route_types
    }


def route_to_handlers(state: RouterState) -> Sequence[str]:
    """Return list of handler nodes to execute in parallel."""
    route_types = state.get("route_types", ["general"])

    handlers = []
    if "iot" in route_types:
        handlers.append("iot_handler")
    # Search routes go to general handler (which has Tavily access)
    if "general" in route_types or "search" in route_types or not handlers:
        handlers.append("general_handler")

    logger.info(f"[route_to_handlers] Will execute: {handlers}")
    return handlers
