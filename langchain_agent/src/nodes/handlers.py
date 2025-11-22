import logging
from typing import Dict, Any
from ..state import RouterState

logger = logging.getLogger('langchain_agent.nodes.handlers')


def iot_handler_node(state: RouterState) -> Dict[str, Any]:
    """Handle IOT device commands."""
    logger.info("[iot_handler] IOT query detected - will use tools")

    response = {
        "handler": "iot",
        "confidence": 0.8
    }

    return {
        "handler_responses": [response]
    }


def general_handler_node(state: RouterState) -> Dict[str, Any]:
    """Handle general queries."""
    logger.info("[general_handler] General query - using LLM")

    response = {
        "handler": "general",
        "confidence": 0.7
    }

    return {
        "handler_responses": [response]
    }
