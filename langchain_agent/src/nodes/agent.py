import logging
from typing import Dict, Any, List, Tuple, TYPE_CHECKING
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, BaseMessage

from ..state import RouterState

if TYPE_CHECKING:
    from ..router_agent_v2 import LangChainRouterAgentV2

logger = logging.getLogger('langchain_agent.nodes.agent')


def has_pending_tool_results(messages: List[BaseMessage]) -> Tuple[bool, List[ToolMessage]]:
    """Check if there are tool results that need to be responded to.

    Returns (True, tool_messages) when:
    1. The most recent message(s) are ToolMessages
    2. These follow an AIMessage that made tool calls

    Returns (False, []) otherwise.
    """
    if not messages:
        return False, []

    recent_tool_messages: List[ToolMessage] = []
    last_ai_with_tools_idx = -1

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            recent_tool_messages.insert(0, msg)
        elif isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                last_ai_with_tools_idx = i
            break
        else:
            break

    if last_ai_with_tools_idx == -1 or not recent_tool_messages:
        return False, []

    return True, recent_tool_messages


def analyze_tool_results(tool_messages: List[ToolMessage]) -> Dict[str, Any]:
    """Analyze tool results to determine success/failure status."""
    results: Dict[str, Any] = {
        "total": len(tool_messages),
        "successful": 0,
        "failed": 0,
        "details": []
    }

    ha_success_indicators = [
        "'response_type': 'action_done'",
        '"response_type": "action_done"',
        "'success': [",
        '"success": [',
    ]
    ha_error_indicators = [
        "'response_type': 'error'",
        '"response_type": "error"',
    ]
    general_error_indicators = ["error", "failed", "not found", "unable", "cannot", "couldn't", "invalid"]

    for msg in tool_messages:
        content = str(msg.content) if msg.content else ""
        content_lower = content.lower()
        tool_name = msg.name if hasattr(msg, 'name') and msg.name else "unknown"

        is_ha_success = any(indicator in content for indicator in ha_success_indicators)
        is_ha_error = any(indicator in content for indicator in ha_error_indicators)
        is_general_error = any(indicator in content_lower for indicator in general_error_indicators)

        if is_ha_success and not is_ha_error:
            is_success = True
        elif is_ha_error:
            is_success = False
        else:
            is_success = not is_general_error

        if is_success:
            results["successful"] += 1
            results["details"].append({
                "tool": tool_name,
                "status": "success",
                "content": msg.content
            })
        else:
            results["failed"] += 1
            results["details"].append({
                "tool": tool_name,
                "status": "failed",
                "content": msg.content
            })

    return results


def create_agent_node(agent_instance: 'LangChainRouterAgentV2'):
    """Create an agent node that handles tool-result continuation.

    After the handler nodes process the initial query, this node only activates
    when there are pending tool results from Home Assistant that need a
    natural language summary.
    """

    def agent_node(state: RouterState) -> Dict[str, Any]:
        messages = state["messages"]

        pending_results, tool_messages = has_pending_tool_results(messages)

        if not pending_results:
            logger.info("[agent] No pending tool results, passing through (handler already responded)")
            return {}

        logger.info("[agent] Processing tool results")
        results_analysis = analyze_tool_results(tool_messages)
        logger.info(f"[agent] Tool results: {results_analysis['successful']} succeeded, {results_analysis['failed']} failed out of {results_analysis['total']}")

        for detail in results_analysis["details"]:
            logger.info(f"[agent]   - {detail['tool']}: {detail['status']}")

        results_summary = "\n".join([
            f"- {d['tool']}: {d['status']} - {d['content']}"
            for d in results_analysis["details"]
        ])

        system_prompt = SystemMessage(content=f"""You are a voice assistant. Generate brief, natural responses based on the ACTUAL tool results below.

TOOL EXECUTION RESULTS:
{results_summary}

CRITICAL: Base your response ONLY on what actually happened according to the tool results above.
- If a tool succeeded, confirm the action
- If a tool failed or returned an error, explain what went wrong
- NEVER claim an action was successful if the tool result shows otherwise

Guidelines:
- Be concise and conversational
- Use natural phrasing ("I've added..." not "I have successfully added...")
- Don't add phrases like "It should now be..." or "You have saved"
- For errors, explain briefly what went wrong

Examples of CORRECT responses:
- Tool succeeded: "I've added apple to your shopping list"
- Tool succeeded: "The bedroom light is on"
- Tool failed: "I couldn't add that item - the list wasn't found"
- Tool failed: "I couldn't turn on the light - device not found"
""")

        messages_with_system = [system_prompt] + messages
        response = agent_instance.chat_device.invoke(messages_with_system)
        return {
            "messages": [response]
        }

    return agent_node
