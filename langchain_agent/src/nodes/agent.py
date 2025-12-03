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
    results = {
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
    """Create an agent node with access to the agent instance's models and tools."""

    def agent_node(state: RouterState) -> Dict[str, Any]:
        """Agent decides whether to call tools or return response."""
        logger.info("[agent] Processing with LLM")

        ha_tools = state.get("tools", [])
        messages = state["messages"]

        pending_results, tool_messages = has_pending_tool_results(messages)

        if pending_results:
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

        logger.info("[agent] No pending tool results, checking if tools needed for new query")
        logger.info(f"[agent] Messages in state: {len(messages)}")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:80].replace('\n', ' ') if msg.content else "(empty)"
            logger.info(f"[agent]   {i+1}. {msg_type}: {content_preview}")

        all_tools = list(ha_tools) + agent_instance.local_tools
        has_tools = len(all_tools) > 0

        if has_tools:
            logger.info(f"[agent] Available tools: {len(ha_tools)} HA + {len(agent_instance.local_tools)} local = {len(all_tools)} total")

            system_prompt = SystemMessage(content="""You are a smart home assistant with access to various tools and functions.

## Conversation Context
You have access to the conversation history. Use it to:
- Understand pronouns like "them", "it", "those" (e.g., "dim them" refers to lights mentioned earlier)
- Answer questions about previous requests (e.g., "what was my last request?" - just look at the history)
- Maintain continuity (e.g., if user said "office lights", then "dim them to 50%" means office lights)

DO NOT call tools for questions about conversation history - just answer from the messages you can see.

## When to Use Tools
Call tools ONLY when you need to:
- Perform an ACTION (turn on/off, set brightness, add to list, etc.)
- Get CURRENT device state that's not in the conversation

## Understanding Context

The user's message may contain device state information in a format like:
- entity_id 'Name' = state
- Examples: "weather.forecast_home 'Forecast Home' = partlycloudy;17.6 °C;83%"
- "light.bedroom 'Bedroom' = on;100%"

If you see this device information, you can use it directly to answer questions about home state without calling a tool.

## Tool Usage Guidelines

### Controlling Lights
For lights, use the 'domain' parameter, NOT 'device_class':
- HassTurnOn({'name': 'bedroom light'}) - turn on by name
- HassTurnOn({'domain': ['light'], 'area': 'bedroom'}) - turn on all lights in an area
- HassTurnOff({'domain': ['light'], 'floor': 'upstairs'}) - turn off all lights on a floor

For setting brightness or color, use HassLightSet:
- HassLightSet({'area': 'office', 'brightness': 50}) - set brightness to 50%
- HassLightSet({'area': 'office', 'color': 'red'}) - set color to red
- HassLightSet({'name': 'bedroom light', 'color': 'amber'}) - set color to amber

IMPORTANT: Always TRY to set colors/brightness with HassLightSet. Don't assume a light can't change color - let Home Assistant determine that.

IMPORTANT: 'light' is NOT a valid device_class. Use 'domain': ['light'] for lights.

### Controlling Switches/Outlets
For switches and outlets, use device_class:
- HassTurnOn({'device_class': ['switch'], 'name': 'fan'})
- HassTurnOn({'device_class': ['outlet'], 'area': 'garage'})

### Shopping Lists and To-Do Lists
- HassListAddItem({'item': 'milk', 'name': 'Shopping List'})
- todo_get_items({'todo_list': 'Shopping List'})

### Weather and Home State
- If weather.* entity data is in the context, use it directly
- If not, use tavily_web_search for weather, news, or current events
- For "state of my home" questions, summarize the device states in the context

### Getting Real-time Data
- Use GetLiveContext when you need current values not in the provided context
- Use HassGetState for specific device states

## Quick Examples
- "Turn on the bedroom light" → HassTurnOn({'name': 'bedroom light'})
- "Turn off all lights" → HassTurnOff({'domain': ['light']})
- "Add milk to shopping list" → HassListAddItem({'item': 'milk', 'name': 'Shopping List'})
- "What's the weather?" → Check for weather.* in context, otherwise use tavily_web_search
- "State of my home" → Summarize device states from context

Always prefer using context information when available, then tools, over generating a text-only response.""")

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
