import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage

from ..state import RouterState
from ..utils import preview_text

logger = logging.getLogger('langchain_agent.nodes.aggregator')


def aggregator_node(state: RouterState) -> Dict[str, Any]:
    """Verify that the handler produced output and log a summary."""
    handler_responses = state.get("handler_responses", [])
    active_handler = state.get("active_handler")
    messages = state.get("messages", [])

    logger.info(f"[aggregator] Active handler: {active_handler}, handler_responses: {len(handler_responses)}")

    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            content_preview = preview_text(str(last_msg.content), 100) if last_msg.content else "(empty)"
            has_tools = bool(hasattr(last_msg, 'tool_calls') and last_msg.tool_calls)
            logger.info(f"[aggregator] Handler output: {content_preview}")
            if has_tools:
                logger.info(f"[aggregator] Handler produced {len(last_msg.tool_calls)} tool call(s)")
        else:
            logger.warning(f"[aggregator] Last message is {type(last_msg).__name__}, expected AIMessage from handler")

    return {}
