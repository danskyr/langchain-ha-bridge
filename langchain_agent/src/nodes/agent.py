import logging
from typing import Dict, Any, TYPE_CHECKING
from langchain_core.messages import SystemMessage, ToolMessage

from ..state import RouterState

if TYPE_CHECKING:
    from ..router_agent_v2 import LangChainRouterAgentV2

logger = logging.getLogger('langchain_agent.nodes.agent')


def create_agent_node(agent_instance: 'LangChainRouterAgentV2'):
    """Create an agent node with access to the agent instance's models and tools."""

    def agent_node(state: RouterState) -> Dict[str, Any]:
        """Agent decides whether to call tools or return response."""
        logger.info("[agent] Processing with LLM")

        ha_tools = state.get("tools", [])
        messages = state["messages"]

        has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)

        if has_tool_results:
            logger.info("[agent] Tool results detected, generating final response")

            system_prompt = SystemMessage(content="""You are a voice assistant. Generate brief, natural responses for completed actions.

Guidelines:
- Be concise and conversational
- Confirm what was done without extra explanation
- Use natural phrasing ("I've added..." not "I have successfully added...")
- Don't add phrases like "It should now be..." or "You have saved"
- For errors, explain briefly what went wrong

Examples:
- "I've added apple to your shopping list"
- "The bedroom light is on"
- "Done"
- "I couldn't find that device"
""")

            messages_with_system = [system_prompt] + messages
            response = agent_instance.chat_device.invoke(messages_with_system)
            return {
                "messages": [response]
            }

        all_tools = list(ha_tools) + agent_instance.local_tools
        has_tools = len(all_tools) > 0

        if has_tools:
            logger.info(f"[agent] Available tools: {len(ha_tools)} HA + {len(agent_instance.local_tools)} local = {len(all_tools)} total")

            system_prompt = SystemMessage(content="""You are a smart home assistant with access to various tools and functions.

IMPORTANT: When the user asks you to perform an action or needs current information, you MUST call the appropriate tool. Do not just describe what you would do - actually call the tool.

Examples:
- "Add milk to shopping list" → Call HassListAddItem tool
- "Turn on the lights" → Call HassTurnOn tool
- "What's the weather today?" → Call tavily_web_search tool
- "What happened in the news yesterday?" → Call tavily_web_search tool

Always prefer using tools over generating a text-only response when a tool is available.""")

            messages_with_system = [system_prompt] + messages

            llm_with_tools = agent_instance.chat_device.bind_tools(all_tools)

            logger.info(f"[agent] Invoking LLM with {len(all_tools)} tools")
            response = llm_with_tools.invoke(messages_with_system)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"[agent] LLM requested {len(response.tool_calls)} tool calls")
                for tc in response.tool_calls:
                    logger.debug(f"[agent] Tool call: {tc.get('name', 'unknown')}")
                return {
                    "messages": [response]
                }
            else:
                logger.warning(f"[agent] LLM did not call tools despite having {len(all_tools)} available")
                logger.debug(f"[agent] Response content: {response.content[:200] if response.content else 'None'}")

        logger.info("[agent] Generating final response without tools")
        response = agent_instance.chat_device.invoke(messages)

        return {
            "messages": [response]
        }

    return agent_node
