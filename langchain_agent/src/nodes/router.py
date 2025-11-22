import logging
from typing import Dict, Any, Sequence
from ..state import RouterState
from ..utils import preview_text

logger = logging.getLogger('langchain_agent.nodes.router')


def router_node(state: RouterState) -> Dict[str, Any]:
    """Determine which handlers should process this query."""
    query = state["query"]
    logger.info(f"[router] Analyzing query: {preview_text(query, 100)}")

    route_types = []

    query_lower = query.lower()

    iot_keywords = [
        "turn", "set", "light", "temperature", "switch", "device", "open", "close",
        "add", "list", "todo", "shopping", "remove", "delete", "complete", "task"
    ]
    if any(keyword in query_lower for keyword in iot_keywords):
        route_types.append("iot")

    search_keywords = ["what", "who", "when", "where", "weather", "news", "search", "find"]
    if any(keyword in query_lower for keyword in search_keywords):
        route_types.append("search")

    if not route_types:
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
    if "general" in route_types or not handlers:
        handlers.append("general_handler")

    logger.info(f"[route_to_handlers] Will execute: {handlers}")
    return handlers
