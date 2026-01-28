import logging
from typing import Dict, Any, TYPE_CHECKING
from langchain_core.messages import SystemMessage

from ..state import RouterState
from ..utils import preview_text

if TYPE_CHECKING:
    from ..router_agent_v2 import LangChainRouterAgentV2

logger = logging.getLogger('langchain_agent.nodes.general_handler')


def create_general_handler_node(agent_instance: 'LangChainRouterAgentV2'):
    """Create a general handler for conversational queries (greetings, help, etc.)."""

    def general_handler_node(state: RouterState) -> Dict[str, Any]:
        logger.info("[general_handler] Processing general query")

        messages = state["messages"]

        system_prompt = SystemMessage(content="""You are a friendly smart home voice assistant. You can help with general conversation, answer questions about what you can do, and respond to greetings.

You can:
- Control smart home devices (lights, switches, media players)
- Search the web for information
- Manage shopping and to-do lists
- Answer general questions

Keep responses brief and conversational, suitable for voice output.""")

        messages_with_system = [system_prompt] + messages
        response = agent_instance.chat_query.invoke(messages_with_system)

        logger.info(f"[general_handler] Response: {preview_text(str(response.content), 100)}")

        return {
            "messages": [response],
            "handler_responses": [{"handler": "general"}],
            "active_handler": "general"
        }

    return general_handler_node
