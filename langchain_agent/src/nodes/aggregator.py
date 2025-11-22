import logging
from typing import Dict, Any
from ..state import RouterState

logger = logging.getLogger('langchain_agent.nodes.aggregator')


def aggregator_node(state: RouterState) -> Dict[str, Any]:
    """Aggregate handler metadata (not adding messages to avoid confusing LLM)."""
    handler_responses = state.get("handler_responses", [])

    logger.info(f"[aggregator] Processed {len(handler_responses)} handler(s)")

    if handler_responses:
        handler_names = [resp.get("handler", "unknown") for resp in handler_responses]
        logger.debug(f"[aggregator] Handlers: {', '.join(handler_names)}")

    return {}
