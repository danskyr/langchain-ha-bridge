import json
import logging
from typing import Dict, Any, List, TYPE_CHECKING
from langchain_core.messages import AIMessage

from ..state import RouterState
from ..utils.validation import validate_tool_call

if TYPE_CHECKING:
    from ..router_agent_v2 import LangChainRouterAgentV2

logger = logging.getLogger('langchain_agent.nodes.validation')

# Known local tools - everything else is assumed to be a Home Assistant tool
LOCAL_TOOLS = {"tavily_web_search"}


def separate_tool_calls(tool_calls: List[Dict[str, Any]]) -> tuple:
    """
    Separate HA tools from local tools.

    Local tools are explicitly listed in LOCAL_TOOLS.
    Everything else is assumed to be a Home Assistant tool.

    Returns:
        Tuple of (ha_tools, local_tools)
    """
    local_tools = [tc for tc in tool_calls if tc.get("name", "") in LOCAL_TOOLS]
    ha_tools = [tc for tc in tool_calls if tc.get("name", "") not in LOCAL_TOOLS]

    logger.info(f"[separate_tool_calls] HA: {len(ha_tools)}, Local: {len(local_tools)}")

    return ha_tools, local_tools


def create_validation_node(agent_instance: 'LangChainRouterAgentV2'):
    """Create a validation node with access to the agent instance's logger."""

    def validation_node(state: RouterState) -> Dict[str, Any]:
        """
        Validate tool calls against their schemas before execution.

        If validation fails:
        - Inject error message back to agent for self-correction
        - Increment validation_attempts counter
        - Loop back to agent (up to max 3 total attempts)
        """
        logger.info("[validation] Validating tool calls")

        messages = state["messages"]
        last_message = messages[-1]
        validation_attempts = state.get("validation_attempts", 1)
        original_query = state.get("query", "")
        ha_tools = state.get("tools", [])

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.info("[validation] No tool calls to validate")
            return {}

        tool_calls = last_message.tool_calls

        ha_tool_calls, local_tool_calls = separate_tool_calls(tool_calls)

        if not ha_tool_calls:
            logger.info("[validation] No HA tool calls to validate")
            return {}

        tool_schemas = {tool.get("function", {}).get("name"): tool for tool in ha_tools}

        validation_errors = []
        for tc in ha_tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_schema = tool_schemas.get(tool_name)

            if not tool_schema:
                logger.warning(f"[validation] No schema found for tool: {tool_name}")
                continue

            # Log the full tool schema for debugging
            logger.info(f"[validation] Tool schema for {tool_name}:\n{json.dumps(tool_schema, indent=2)}")
            logger.info(f"[validation] Tool args provided: {json.dumps(tool_args, indent=2)}")

            error_msg = validate_tool_call(
                tool_name=tool_name,
                args=tool_args,
                tool_schema=tool_schema,
                original_query=original_query,
                logger=agent_instance.logger
            )

            if error_msg:
                validation_errors.append(error_msg)

        if not validation_errors:
            logger.info("[validation] ✓ All tool calls valid")
            return {}

        logger.warning(f"[validation] ✗ {len(validation_errors)} tool call(s) failed validation")
        logger.info(f"[validation] Attempt {validation_attempts} of 3")

        if validation_attempts >= 3:
            logger.error("[validation] Max validation attempts (3) reached. Giving up.")
            error_summary = "\n\n".join(validation_errors)
            final_error = (
                f"After 3 attempts, I was unable to generate valid tool calls for your request.\n\n"
                f"Final errors:\n{error_summary}"
            )

            return {
                "final_response": final_error,
                "validation_attempts": validation_attempts + 1
            }

        logger.info("[validation] Injecting validation errors and retrying")

        combined_error = "\n\n---\n\n".join(validation_errors)
        error_message = AIMessage(content=combined_error)

        return {
            "messages": [error_message],
            "validation_attempts": validation_attempts + 1
        }

    return validation_node


def validation_decision(state: RouterState) -> str:
    """
    Decide where to route after validation.

    Returns:
        "retry" - Validation failed, loop back to agent
        "ha_tools" - Valid HA tools, return to Home Assistant
        "local_tools" - Valid local tools, execute locally
        "formatter" - No tools or max retries hit
    """
    messages = state["messages"]
    last_message = messages[-1]
    validation_attempts = state.get("validation_attempts", 1)

    logger.info(f"[validation_decision] Current attempt: {validation_attempts}")
    logger.info(f"[validation_decision] Last message type: {type(last_message).__name__}")
    logger.info(f"[validation_decision] Has tool_calls attr: {hasattr(last_message, 'tool_calls')}")
    if hasattr(last_message, "tool_calls"):
        logger.info(f"[validation_decision] tool_calls value: {last_message.tool_calls}")

    if state.get("final_response"):
        logger.info("[validation_decision] Max retries hit, going to formatter")
        return "formatter"

    if validation_attempts > 1:
        has_real_tool_calls = (
            hasattr(last_message, "tool_calls") and
            last_message.tool_calls and
            len(last_message.tool_calls) > 0
        )
        if isinstance(last_message, AIMessage) and not has_real_tool_calls:
            logger.info(f"[validation_decision] Validation error detected, retry attempt {validation_attempts}/3")
            return "retry"

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.info("[validation_decision] No tool calls, going to formatter")
        return "formatter"

    tool_calls = last_message.tool_calls
    ha_tools, local_tools = separate_tool_calls(tool_calls)

    if ha_tools and local_tools:
        logger.warning("[validation_decision] Mixed HA and local tool calls, prioritizing HA")
        return "ha_tools"
    elif ha_tools:
        logger.info(f"[validation_decision] {len(ha_tools)} valid HA tool call(s), returning to Home Assistant")
        return "ha_tools"
    elif local_tools:
        logger.info(f"[validation_decision] {len(local_tools)} valid local tool call(s), executing locally")
        return "local_tools"
    else:
        logger.info("[validation_decision] No valid tool calls, going to formatter")
        return "formatter"
