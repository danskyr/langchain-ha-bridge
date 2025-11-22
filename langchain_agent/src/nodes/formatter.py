import logging
from typing import Dict, Any
from ..state import RouterState

logger = logging.getLogger('langchain_agent.nodes.formatter')


def formatter_node(state: RouterState) -> Dict[str, Any]:
    """Format the final response for the user."""
    logger.info("[formatter] Formatting response")

    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "content"):
        final_response = last_message.content
    else:
        final_response = str(last_message)

    if len(final_response) > 200:
        logger.info("[formatter] Response is long, consider summarizing")

    # Determine if conversation should continue based on response content
    # Continue if response ends with question mark (expecting user input)
    continue_conversation = final_response.strip().endswith(("?", ";", "？"))
    logger.info(f"[formatter] continue_conversation: {continue_conversation}")

    return {
        "final_response": final_response,
        "continue_conversation": continue_conversation
    }
