import logging
import os
import uuid
import json
from typing import TypedDict, Optional, Dict, Any, List, Sequence, Annotated
import operator

from dotenv import load_dotenv
import jsonschema
from jsonschema import Draft7Validator, ValidationError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

dotenv_result = load_dotenv()
if not dotenv_result:
    logging.warning(".env file not found or could not be loaded.")

module_logger = logging.getLogger('langchain_agent')
log_level = os.getenv('LANGCHAIN_LOG_LEVEL', 'INFO')
module_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def create_tavily_tool():
    """Create a Tavily search tool if API key is available."""
    if not TAVILY_API_KEY:
        return None

    tavily_search = TavilySearch(
        tavily_api_key=TAVILY_API_KEY,
        max_results=5,
        topic="general"
    )

    @tool
    def tavily_web_search(query: str) -> str:
        """Search the web for current information about events, news, weather, facts, and real-time data.

        Use this tool when you need to find:
        - Current events and news
        - Weather information
        - Recent facts and data
        - Information that changes over time
        - Things that happened recently

        Args:
            query: The search query to look up

        Returns:
            Formatted search results with relevant information
        """
        try:
            module_logger.info(f"[tavily_tool] Searching: {query[:100]}")
            response = tavily_search.invoke({"query": query})
            results = response.get("results", [])

            if not results:
                return "No search results found for that query."

            # Format top 3 results
            formatted = "Search results:\n\n"
            for i, result in enumerate(results[:3], 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "")
                formatted += f"{i}. {title}\n{content}\n"
                if url:
                    formatted += f"Source: {url}\n"
                formatted += "\n"

            return formatted.strip()
        except Exception as e:
            module_logger.error(f"[tavily_tool] Error: {e}")
            return f"Search error: {str(e)}"

    return tavily_web_search


def preview_text(text: str, max_len: int = 600) -> str:
    """Truncate text for logging, showing start and end for long content."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    half = (max_len - 5) // 2
    return f"{text[:half]} ... {text[-half:]}"


def format_validation_error_for_agent(
    tool_name: str,
    args: Dict[str, Any],
    error: ValidationError,
    original_query: str
) -> str:
    """
    Convert jsonschema ValidationError into an agent-friendly error message.

    Args:
        tool_name: Name of the tool that failed validation
        args: The arguments that were provided
        error: The ValidationError from jsonschema
        original_query: The user's original query for context

    Returns:
        A formatted error message that helps the LLM correct its mistake
    """
    # Build the path to the problematic field
    if error.path:
        field_path = ".".join(str(p) for p in error.path)
        field_name = list(error.path)[-1] if error.path else "unknown"
    else:
        field_path = "root"
        field_name = "root"

    # Get the invalid value
    invalid_value = error.instance

    # Build a contextual error message based on the validator type
    if error.validator == "enum":
        # Special handling for enum errors - show allowed values clearly
        allowed_values = error.validator_value
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error: Parameter '{field_name}' has invalid value: {json.dumps(invalid_value)}\n"
            f"Allowed values: {json.dumps(allowed_values)}\n\n"
            f"The value you provided ({json.dumps(invalid_value)}) is not in the list of allowed values.\n"
            f"Please call the tool again with a valid value from the allowed list, "
            f"or omit this parameter if it's not required."
        )
    elif error.validator == "type":
        expected_type = error.validator_value
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error: Parameter '{field_name}' has incorrect type.\n"
            f"Expected type: {expected_type}\n"
            f"Actual value: {json.dumps(invalid_value)}\n\n"
            f"Please call the tool again with the correct type."
        )
    elif error.validator == "required":
        missing_props = error.validator_value
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error: Required parameter(s) missing: {missing_props}\n\n"
            f"Please call the tool again with all required parameters."
        )
    else:
        # Generic error for other validation failures
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error at '{field_path}': {error.message}\n"
            f"Provided value: {json.dumps(invalid_value)}\n\n"
            f"Please call the tool again with valid parameters."
        )

    return error_msg


def validate_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    tool_schema: Dict[str, Any],
    original_query: str,
    logger: logging.Logger
) -> Optional[str]:
    """
    Validate a tool call's arguments against its JSON Schema.

    Args:
        tool_name: Name of the tool being called
        args: Arguments provided to the tool
        tool_schema: The OpenAI function format tool schema
        original_query: User's original query for error context
        logger: Logger instance

    Returns:
        None if valid, or an agent-friendly error message if invalid
    """
    try:
        # Extract the parameters schema (JSON Schema format)
        parameters_schema = tool_schema.get("function", {}).get("parameters", {})

        if not parameters_schema:
            logger.warning(f"[validate_tool_call] No parameters schema found for {tool_name}")
            return None

        # Validate using jsonschema
        jsonschema.validate(instance=args, schema=parameters_schema)

        # Validation passed
        logger.info(f"[validate_tool_call] ✓ {tool_name} arguments are valid")
        return None

    except ValidationError as e:
        # Log technical error details
        logger.warning(f"[validate_tool_call] ✗ {tool_name} validation failed")
        logger.warning(f"[validate_tool_call]   Technical error: {e.message}")
        logger.warning(f"[validate_tool_call]   Path: {list(e.path)}")
        logger.warning(f"[validate_tool_call]   Validator: {e.validator}")
        logger.warning(f"[validate_tool_call]   Invalid value: {e.instance}")
        if e.validator == "enum":
            logger.warning(f"[validate_tool_call]   Allowed values: {e.validator_value}")

        # Format agent-friendly error message
        friendly_error = format_validation_error_for_agent(
            tool_name=tool_name,
            args=args,
            error=e,
            original_query=original_query
        )

        logger.info(f"[validate_tool_call]   Formatted error for agent:\n{friendly_error}")

        return friendly_error
    except Exception as e:
        logger.error(f"[validate_tool_call] Unexpected error validating {tool_name}: {e}")
        return None  # Don't block on unexpected errors


class RouterState(TypedDict):
    """State for the router agent with automatic message merging."""
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    route_types: List[str]
    handler_responses: Annotated[List[Dict[str, Any]], operator.add]
    final_response: Optional[str]
    tools: Optional[List[Dict[str, Any]]]
    validation_attempts: int  # Track number of validation attempts (max 3)


class LangChainRouterAgentV2:
    """
    Improved router agent with proper LangGraph architecture.

    Features:
    - Checkpointing for distributed tool execution
    - Parallel handler execution
    - Clean state management with reducers
    - Integrated tool support with interrupts
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger('langchain_agent.LangChainRouterAgentV2')

        # Initialize models
        self.chat_router = ChatOllama(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0
        )
        self.chat_device = ChatOllama(
            model="qwen2.5:3b",
            base_url="http://localhost:11434",
            temperature=0
        )
        self.chat_query = ChatOllama(
            model="qwen2.5:3b",

            base_url="http://localhost:11434",
            temperature=0
        )

        # Initialize local tools
        self.local_tools = []
        tavily_tool = create_tavily_tool()
        if tavily_tool:
            self.local_tools.append(tavily_tool)
            self.logger.info(f"Initialized {len(self.local_tools)} local tool(s): Tavily search")
        else:
            self.logger.warning("Tavily API key not found - web search will not be available")

        # Build graph with checkpointing
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the state graph with checkpointing and interrupts."""
        workflow = StateGraph(RouterState)

        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("iot_handler", self._iot_handler_node)
        # search_handler removed - now using tavily_web_search tool instead
        workflow.add_node("general_handler", self._general_handler_node)
        workflow.add_node("aggregator", self._aggregator_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tool_call_validation", self._validation_node)
        workflow.add_node("formatter", self._formatter_node)

        # Add local tools node for executing tools like Tavily search
        if self.local_tools:
            workflow.add_node("local_tools", ToolNode(self.local_tools))

        # Define edges
        workflow.add_edge(START, "router")

        # Conditional routing to handlers (can be parallel)
        workflow.add_conditional_edges(
            "router",
            self._route_to_handlers,
            ["iot_handler", "general_handler"]
        )

        # All handlers converge to aggregator
        workflow.add_edge("iot_handler", "aggregator")
        workflow.add_edge("general_handler", "aggregator")

        # Aggregator to agent
        workflow.add_edge("aggregator", "agent")

        # Agent goes to tool call validation first
        workflow.add_edge("agent", "tool_call_validation")

        # Tool call validation decides what to do next
        validation_edge_map = {
            "retry": "agent",  # Validation failed, retry
            "ha_tools": END,  # Valid HA tools, return to Home Assistant
            "formatter": "formatter",  # No tools or max retries hit
        }

        # Add local_tools path if we have local tools
        if self.local_tools:
            validation_edge_map["local_tools"] = "local_tools"
            # Local tools execute and loop back to agent
            workflow.add_edge("local_tools", "agent")

        workflow.add_conditional_edges(
            "tool_call_validation",
            self._validation_decision,
            validation_edge_map
        )

        workflow.add_edge("formatter", END)

        # Compile with checkpointing
        return workflow.compile(checkpointer=self.checkpointer)

    def _router_node(self, state: RouterState) -> Dict[str, Any]:
        """Determine which handlers should process this query."""
        query = state["query"]
        self.logger.info(f"[router] Analyzing query: {preview_text(query, 100)}")

        route_types = []

        # Simple keyword-based routing (can be enhanced with LLM)
        query_lower = query.lower()

        # Check for IOT commands (including todo/shopping list operations)
        iot_keywords = [
            "turn", "set", "light", "temperature", "switch", "device", "open", "close",
            "add", "list", "todo", "shopping", "remove", "delete", "complete", "task"
        ]
        if any(keyword in query_lower for keyword in iot_keywords):
            route_types.append("iot")

        # Check for search queries
        search_keywords = ["what", "who", "when", "where", "weather", "news", "search", "find"]
        if any(keyword in query_lower for keyword in search_keywords):
            route_types.append("search")

        # Default to general
        if not route_types:
            route_types.append("general")

        self.logger.info(f"[router] Routes selected: {route_types}")

        return {
            "route_types": route_types
        }

    def _route_to_handlers(self, state: RouterState) -> Sequence[str]:
        """Return list of handler nodes to execute in parallel."""
        route_types = state.get("route_types", ["general"])

        handlers = []
        if "iot" in route_types:
            handlers.append("iot_handler")
        # search_handler removed - agent will use tavily_web_search tool instead
        if "general" in route_types or not handlers:
            handlers.append("general_handler")

        self.logger.info(f"[route_to_handlers] Will execute: {handlers}")
        return handlers

    def _iot_handler_node(self, state: RouterState) -> Dict[str, Any]:
        """Handle IOT device commands."""
        self.logger.info("[iot_handler] IOT query detected - will use tools")

        # Don't add content - let the agent with tools handle it
        # Just track that this handler ran
        response = {
            "handler": "iot",
            "confidence": 0.8
        }

        return {
            "handler_responses": [response]
        }

    def _search_handler_node(self, state: RouterState) -> Dict[str, Any]:
        """Handle search queries."""
        self.logger.info("[search_handler] Processing search query")

        query = state["query"]

        if self.tavily_search:
            try:
                self.logger.info(f"[search_handler] Searching: {preview_text(query, 100)}")
                tavily_response = self.tavily_search.invoke({"query": query})
                results = tavily_response.get("results", [])

                if results:
                    top_result = results[0]
                    content = f"Search result: {top_result.get('content', '')}"
                else:
                    content = "No search results found"

                response = {
                    "handler": "search",
                    "confidence": 0.9,
                    "content": content
                }
            except Exception as e:
                self.logger.error(f"[search_handler] Error: {e}")
                response = {
                    "handler": "search",
                    "confidence": 0.3,
                    "content": f"Search error: {str(e)}"
                }
        else:
            response = {
                "handler": "search",
                "confidence": 0.1,
                "content": "Search not available (Tavily API key not set)"
            }

        return {
            "handler_responses": [response]
        }

    def _general_handler_node(self, state: RouterState) -> Dict[str, Any]:
        """Handle general queries."""
        self.logger.info("[general_handler] General query - using LLM")

        # Don't add content - let the agent handle it
        response = {
            "handler": "general",
            "confidence": 0.7
        }

        return {
            "handler_responses": [response]
        }

    def _aggregator_node(self, state: RouterState) -> Dict[str, Any]:
        """Aggregate handler metadata (not adding messages to avoid confusing LLM)."""
        handler_responses = state.get("handler_responses", [])

        self.logger.info(f"[aggregator] Processed {len(handler_responses)} handler(s)")

        # Log which handlers ran
        if handler_responses:
            handler_names = [resp.get("handler", "unknown") for resp in handler_responses]
            self.logger.debug(f"[aggregator] Handlers: {', '.join(handler_names)}")

        # Don't add any messages - let the agent be first to respond
        # This prevents confusing the LLM with handler-generated content
        return {}

    def _agent_node(self, state: RouterState) -> Dict[str, Any]:
        """Agent decides whether to call tools or return response."""
        self.logger.info("[agent] Processing with LLM")

        ha_tools = state.get("tools", [])
        messages = state["messages"]

        # Check if we have ToolMessages (indicating tool results from HA)
        has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)

        if has_tool_results:
            # We have tool results - generate final natural language response
            self.logger.info("[agent] Tool results detected, generating final response")

            # Add concise system prompt for voice assistant
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
            response = self.chat_device.invoke(messages_with_system)
            return {
                "messages": [response]
            }

        # Combine HA tools with local tools
        all_tools = list(ha_tools) + self.local_tools
        has_tools = len(all_tools) > 0

        # If tools are available, use LLM with tools
        if has_tools:
            self.logger.info(f"[agent] Available tools: {len(ha_tools)} HA + {len(self.local_tools)} local = {len(all_tools)} total")

            # Add system prompt instructing LLM to use tools
            system_prompt = SystemMessage(content="""You are a smart home assistant with access to various tools and functions.

IMPORTANT: When the user asks you to perform an action or needs current information, you MUST call the appropriate tool. Do not just describe what you would do - actually call the tool.

Examples:
- "Add milk to shopping list" → Call HassListAddItem tool
- "Turn on the lights" → Call HassTurnOn tool
- "What's the weather today?" → Call tavily_web_search tool
- "What happened in the news yesterday?" → Call tavily_web_search tool

Always prefer using tools over generating a text-only response when a tool is available.""")

            # Prepend system message (only if not already present)
            messages_with_system = [system_prompt] + messages

            # Bind all tools to the device model
            llm_with_tools = self.chat_device.bind_tools(all_tools)

            self.logger.info(f"[agent] Invoking LLM with {len(all_tools)} tools")
            response = llm_with_tools.invoke(messages_with_system)

            # Check if LLM wants to call tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.logger.info(f"[agent] LLM requested {len(response.tool_calls)} tool calls")
                for tc in response.tool_calls:
                    self.logger.debug(f"[agent] Tool call: {tc.get('name', 'unknown')}")
                return {
                    "messages": [response]
                }
            else:
                # Log if we expected tool calls but didn't get them
                self.logger.warning(f"[agent] LLM did not call tools despite having {len(all_tools)} available")
                self.logger.debug(f"[agent] Response content: {response.content[:200] if response.content else 'None'}")

        # No tools or no tool calls - use regular LLM
        self.logger.info("[agent] Generating final response without tools")
        response = self.chat_device.invoke(messages)

        return {
            "messages": [response]
        }

    def _validation_node(self, state: RouterState) -> Dict[str, Any]:
        """
        Validate tool calls against their schemas before execution.

        If validation fails:
        - Inject error message back to agent for self-correction
        - Increment validation_attempts counter
        - Loop back to agent (up to max 3 total attempts)
        """
        self.logger.info("[validation] Validating tool calls")

        messages = state["messages"]
        last_message = messages[-1]
        validation_attempts = state.get("validation_attempts", 1)
        original_query = state.get("query", "")
        ha_tools = state.get("tools", [])

        # Check if we have tool calls to validate
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            self.logger.info("[validation] No tool calls to validate")
            return {}

        tool_calls = last_message.tool_calls

        # Separate HA and local tool calls
        ha_tool_calls, local_tool_calls = self._separate_tool_calls(tool_calls)

        # Only validate HA tools (local tools are our own, assumed valid)
        if not ha_tool_calls:
            self.logger.info("[validation] No HA tool calls to validate")
            return {}

        # Build tool schema lookup
        tool_schemas = {tool.get("function", {}).get("name"): tool for tool in ha_tools}

        # Validate each HA tool call
        validation_errors = []
        for tc in ha_tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_schema = tool_schemas.get(tool_name)

            if not tool_schema:
                self.logger.warning(f"[validation] No schema found for tool: {tool_name}")
                continue

            # Validate
            error_msg = validate_tool_call(
                tool_name=tool_name,
                args=tool_args,
                tool_schema=tool_schema,
                original_query=original_query,
                logger=self.logger
            )

            if error_msg:
                validation_errors.append(error_msg)

        # If all valid, continue
        if not validation_errors:
            self.logger.info("[validation] ✓ All tool calls valid")
            return {}

        # Validation failed - check retry limit
        self.logger.warning(f"[validation] ✗ {len(validation_errors)} tool call(s) failed validation")
        self.logger.info(f"[validation] Attempt {validation_attempts} of 3")

        if validation_attempts >= 3:
            # Hit max retries - give up
            self.logger.error("[validation] Max validation attempts (3) reached. Giving up.")
            error_summary = "\n\n".join(validation_errors)
            final_error = (
                f"After 3 attempts, I was unable to generate valid tool calls for your request.\n\n"
                f"Final errors:\n{error_summary}"
            )

            # Add error as final response and return to formatter
            return {
                "final_response": final_error,
                "validation_attempts": validation_attempts + 1
            }

        # Under retry limit - inject error and loop back to agent
        self.logger.info("[validation] Injecting validation errors and retrying")

        # Create ToolMessage with the validation errors
        combined_error = "\n\n---\n\n".join(validation_errors)
        error_message = AIMessage(content=combined_error)

        return {
            "messages": [error_message],
            "validation_attempts": validation_attempts + 1
        }

    def _validation_decision(self, state: RouterState) -> str:
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

        self.logger.info(f"[validation_decision] Current attempt: {validation_attempts}")
        self.logger.info(f"[validation_decision] Last message type: {type(last_message).__name__}")
        self.logger.info(f"[validation_decision] Has tool_calls attr: {hasattr(last_message, 'tool_calls')}")
        if hasattr(last_message, "tool_calls"):
            self.logger.info(f"[validation_decision] tool_calls value: {last_message.tool_calls}")

        # Check if max retries hit (final_response set by validation node)
        if state.get("final_response"):
            self.logger.info("[validation_decision] Max retries hit, going to formatter")
            return "formatter"

        # Check if we need to retry (validation counter was incremented)
        # The validation node increments the counter when it injects an error message
        if validation_attempts > 1:
            # We've done at least one validation pass and may need to retry
            # Check if the last message is an error message (AIMessage without real tool_calls)
            has_real_tool_calls = (
                hasattr(last_message, "tool_calls") and
                last_message.tool_calls and
                len(last_message.tool_calls) > 0
            )
            if isinstance(last_message, AIMessage) and not has_real_tool_calls:
                # This is a validation error message, agent should retry
                self.logger.info(f"[validation_decision] Validation error detected, retry attempt {validation_attempts}/3")
                return "retry"

        # Check for tool calls (these would be new/corrected tool calls from the agent)
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            self.logger.info("[validation_decision] No tool calls, going to formatter")
            return "formatter"

        # We have tool calls - separate and route them
        tool_calls = last_message.tool_calls
        ha_tools, local_tools = self._separate_tool_calls(tool_calls)

        if ha_tools and local_tools:
            # Mixed calls - prioritize HA tools
            self.logger.warning("[validation_decision] Mixed HA and local tool calls, prioritizing HA")
            return "ha_tools"
        elif ha_tools:
            self.logger.info(f"[validation_decision] {len(ha_tools)} valid HA tool call(s), returning to Home Assistant")
            return "ha_tools"
        elif local_tools:
            self.logger.info(f"[validation_decision] {len(local_tools)} valid local tool call(s), executing locally")
            return "local_tools"
        else:
            self.logger.info("[validation_decision] No valid tool calls, going to formatter")
            return "formatter"

    def _separate_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> tuple:
        """
        Separate HA tools from local tools by name prefix.

        HA tools start with "Hass" (e.g., HassListAddItem, HassTurnOn)
        Local tools are our tools (e.g., tavily_web_search)

        Returns:
            Tuple of (ha_tools, local_tools)
        """
        ha_tools = [tc for tc in tool_calls if tc.get("name", "").startswith("Hass")]
        local_tools = [tc for tc in tool_calls if not tc.get("name", "").startswith("Hass")]

        self.logger.info(f"[separate_tool_calls] HA: {len(ha_tools)}, Local: {len(local_tools)}")

        return ha_tools, local_tools

    def _formatter_node(self, state: RouterState) -> Dict[str, Any]:
        """Format the final response for the user."""
        self.logger.info("[formatter] Formatting response")

        messages = state["messages"]
        last_message = messages[-1]

        # Extract content from last message
        if hasattr(last_message, "content"):
            final_response = last_message.content
        else:
            final_response = str(last_message)

        # Simple formatting for voice assistant
        if len(final_response) > 200:
            # Could add summarization here
            self.logger.info("[formatter] Response is long, consider summarizing")

        return {
            "final_response": final_response
        }

    async def process(
        self,
        query: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Process a query with the graph, handling both initial and continuation calls.

        Args:
            query: The user's query
            tools: Available tools from Home Assistant
            conversation_id: Thread ID for resuming conversation
            tool_results: Results from previously requested tool calls

        Returns:
            Dict with either tool_calls or final response
        """
        # Create or resume thread
        thread_id = conversation_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        self.logger.info(f"[process] Thread: {thread_id[:8]}... | Tool results: {bool(tool_results)}")

        if tool_results:
            # Continue from checkpoint with tool results
            self.logger.info(f"[process] Resuming with {len(tool_results)} tool results")

            # Convert tool results to ToolMessages
            tool_messages = []
            for result in tool_results:
                tool_messages.append(
                    ToolMessage(
                        content=str(result.get("result", "")),
                        tool_call_id=result.get("tool_call_id", "")
                    )
                )

            # Resume graph execution with tool results
            # Reset validation attempts since we're starting a new phase (processing results)
            state_update = {
                "messages": tool_messages,
                "validation_attempts": 1  # Reset counter for tool results phase
            }

            result = await self.graph.ainvoke(state_update, config)
        else:
            # Start new execution
            self.logger.info(f"[process] Starting new conversation")

            initial_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "route_types": [],
                "handler_responses": [],
                "final_response": None,
                "tools": tools,
                "validation_attempts": 1
            }

            result = await self.graph.ainvoke(initial_state, config)

        # Check what the graph returned
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]

            # Check if we need tool execution
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                self.logger.info(f"[process] Returning {len(last_message.tool_calls)} tool calls")

                # Get the original tool definitions for comparison
                ha_tools = result.get("tools", [])
                tool_defs_by_name = {tool.get("function", {}).get("name"): tool for tool in ha_tools}

                # Format tool calls for Home Assistant
                tool_calls = []
                for i, tc in enumerate(last_message.tool_calls, 1):
                    formatted_tc = {
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tc["name"],
                        "args": {k: v for k, v in tc["args"].items() if v is not None}
                    }
                    tool_calls.append(formatted_tc)

                    # Log each tool call in detail
                    self.logger.info(f"[process]   Tool call #{i}:")
                    self.logger.info(f"[process]     - Name: {formatted_tc['name']}")
                    self.logger.info(f"[process]     - ID: {formatted_tc['id']}")
                    self.logger.info(f"[process]     - Args: {formatted_tc['args']}")

                    # Log the original tool definition for comparison
                    # tool_def = tool_defs_by_name.get(formatted_tc['name'])
                    # if tool_def:
                    #     self.logger.info(f"[process]     - Original tool definition:")
                    #     # Pretty print the tool definition
                    #     tool_def_str = json.dumps(tool_def, indent=6)
                    #     for line in tool_def_str.split('\n'):
                    #         self.logger.info(f"[process]       {line}")
                    # else:
                    #     self.logger.info(f"[process]     - Original tool definition: Not found (local tool?)")

                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "conversation_id": thread_id
                }

        # Final response
        final_response = result.get("final_response", "No response generated")
        self.logger.info(f"[process] Returning final response: {preview_text(final_response, 100)}")

        return {
            "type": "response",
            "response": final_response
        }


if __name__ == "__main__":
    import asyncio

    agent = LangChainRouterAgentV2()
    print("Welcome to the LangChain Router Agent V2! 👋")
    print("Type your query, or 'exit' to quit.")

    async def run_cli():
        try:
            while True:
                input_query = input("You: ")
                if not input_query.strip():
                    continue
                if input_query.strip().lower() in ("exit", "quit"):
                    print("Goodbye! 👋")
                    break

                response = await agent.process(input_query)
                print(f"Agent: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")

    asyncio.run(run_cli())
