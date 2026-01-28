import logging
from typing import Dict, Any, TYPE_CHECKING
from langchain_core.messages import SystemMessage

from ..state import RouterState

if TYPE_CHECKING:
    from ..router_agent_v2 import LangChainRouterAgentV2

logger = logging.getLogger('langchain_agent.nodes.iot_handler')


def create_iot_handler_node(agent_instance: 'LangChainRouterAgentV2'):
    """Create an IOT handler node that binds HA tools and invokes LLM for device control."""

    def iot_handler_node(state: RouterState) -> Dict[str, Any]:
        logger.info("[iot_handler] Processing IOT command")

        ha_tools = state.get("tools", [])
        messages = state["messages"]

        all_tools = list(ha_tools) + agent_instance.local_tools
        has_tools = len(all_tools) > 0

        if not has_tools:
            logger.warning("[iot_handler] No tools available for IOT command")
            response = agent_instance.chat_device.invoke(messages)
            return {
                "messages": [response],
                "handler_responses": [{"handler": "iot", "had_tools": False}],
                "active_handler": "iot"
            }

        system_prompt = SystemMessage(content="""You are a smart home assistant with access to various tools and functions.

## Conversation Context
You have access to the conversation history. Use it to:
- Understand pronouns like "them", "it", "those" (e.g., "dim them" refers to lights mentioned earlier)
- Answer questions about previous requests (e.g., "what was my last request?" - just look at the history)
- Maintain continuity (e.g., if user said "office lights", then "dim them to 50%" means office lights)

CRITICAL: For follow-up commands like "turn it off", "dim them", etc., look at your PREVIOUS SUCCESSFUL tool call and use the SAME arguments. Example:
- If you successfully called HassTurnOn({'area': 'office', 'domain': ['light']})
- And user says "turn it off"
- Call HassTurnOff({'area': 'office', 'domain': ['light']}) with the SAME area/domain

DO NOT invent new device names. Reuse the exact arguments that worked before.

## When to Ask for Clarification
If you're unsure which device the user means, ASK instead of guessing:
- "Which light would you like me to turn off - the office or bedroom?"
- "I see multiple lights in the living room. Do you mean the Reading Light or the Mood Lamp?"

Ask for clarification when:
- The user says "it" or "them" but there's no clear reference in conversation history
- Multiple devices could match the request
- You're not confident about which device to control

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

        logger.info(f"[iot_handler] Invoking LLM with {len(all_tools)} tools ({len(ha_tools)} HA + {len(agent_instance.local_tools)} local)")
        response = llm_with_tools.invoke(messages_with_system)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"[iot_handler] LLM requested {len(response.tool_calls)} tool calls")
            for tc in response.tool_calls:
                logger.debug(f"[iot_handler] Tool call: {tc.get('name', 'unknown')}")
        else:
            logger.info("[iot_handler] LLM responded without tool calls")

        return {
            "messages": [response],
            "handler_responses": [{"handler": "iot", "had_tools": True, "tool_count": len(all_tools)}],
            "active_handler": "iot"
        }

    return iot_handler_node
