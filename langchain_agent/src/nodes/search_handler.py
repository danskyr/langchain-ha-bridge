import logging
from typing import Dict, Any, TYPE_CHECKING
from langchain_core.messages import SystemMessage

from ..state import RouterState
from ..utils import preview_text

if TYPE_CHECKING:
    from ..router_agent_v2 import LangChainRouterAgentV2

logger = logging.getLogger('langchain_agent.nodes.search_handler')


def create_search_handler_node(agent_instance: 'LangChainRouterAgentV2'):
    """Create a search handler that calls Tavily directly and synthesizes results."""

    def search_handler_node(state: RouterState) -> Dict[str, Any]:
        logger.info("[search_handler] Processing search query")

        query = state["query"]
        messages = state["messages"]

        if not agent_instance.local_tools:
            logger.warning("[search_handler] No Tavily tool available, falling back to LLM")
            response = agent_instance.chat_query.invoke(messages)
            return {
                "messages": [response],
                "handler_responses": [{"handler": "search", "used_tavily": False}],
                "active_handler": "search"
            }

        tavily_tool = agent_instance.local_tools[0]

        try:
            logger.info(f"[search_handler] Searching for: {preview_text(query, 100)}")
            search_results = tavily_tool.invoke(query)
            logger.info(f"[search_handler] Got results: {preview_text(str(search_results), 200)}")
        except Exception as e:
            logger.error(f"[search_handler] Tavily search failed: {e}")
            response = agent_instance.chat_query.invoke(messages)
            return {
                "messages": [response],
                "handler_responses": [{"handler": "search", "used_tavily": False, "error": str(e)}],
                "active_handler": "search"
            }

        synthesis_prompt = SystemMessage(content=f"""You are a helpful voice assistant. Answer the user's question using the search results below. Be concise and conversational.

SEARCH RESULTS:
{search_results}

Guidelines:
- Answer directly and naturally
- Cite sources only if the user asks for them
- If the results don't fully answer the question, say what you found and note what's missing
- Keep responses brief and suitable for voice output""")

        synthesis_messages = [synthesis_prompt] + messages
        response = agent_instance.chat_query.invoke(synthesis_messages)

        logger.info(f"[search_handler] Synthesized response: {preview_text(str(response.content), 100)}")

        return {
            "messages": [response],
            "handler_responses": [{"handler": "search", "used_tavily": True}],
            "active_handler": "search"
        }

    return search_handler_node
