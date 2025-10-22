import logging
import os
import uuid
from typing import TypedDict, Optional, Dict, Any, List, Sequence, Annotated
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
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


def preview_text(text: str, max_len: int = 600) -> str:
    """Truncate text for logging, showing start and end for long content."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    half = (max_len - 5) // 2
    return f"{text[:half]} ... {text[-half:]}"


class RouterState(TypedDict):
    """State for the router agent with automatic message merging."""
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    route_types: List[str]
    handler_responses: Annotated[List[Dict[str, Any]], operator.add]
    final_response: Optional[str]
    tools: Optional[List[Dict[str, Any]]]


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

        # Tavily search (optional)
        if TAVILY_API_KEY:
            self.tavily_search = TavilySearch(
                tavily_api_key=TAVILY_API_KEY,
                max_results=5,
                topic="general"
            )
        else:
            self.tavily_search = None

        # Build graph with checkpointing
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the state graph with checkpointing and interrupts."""
        workflow = StateGraph(RouterState)

        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("iot_handler", self._iot_handler_node)
        workflow.add_node("search_handler", self._search_handler_node)
        workflow.add_node("general_handler", self._general_handler_node)
        workflow.add_node("aggregator", self._aggregator_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("formatter", self._formatter_node)

        # Define edges
        workflow.add_edge(START, "router")

        # Conditional routing to handlers (can be parallel)
        workflow.add_conditional_edges(
            "router",
            self._route_to_handlers,
            ["iot_handler", "search_handler", "general_handler"]
        )

        # All handlers converge to aggregator
        workflow.add_edge("iot_handler", "aggregator")
        workflow.add_edge("search_handler", "aggregator")
        workflow.add_edge("general_handler", "aggregator")

        # Aggregator to agent
        workflow.add_edge("aggregator", "agent")

        # Agent decides: tools or formatter
        workflow.add_conditional_edges(
            "agent",
            self._should_continue_to_tools,
            {
                "tools": END,  # Return to caller for tool execution
                "formatter": "formatter"
            }
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
        if "search" in route_types:
            handlers.append("search_handler")
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

        tools = state.get("tools", [])
        messages = state["messages"]

        # If tools are available, use LLM with tools
        if tools:
            # Add system prompt instructing LLM to use tools
            system_prompt = SystemMessage(content="""You are a smart home assistant with access to various tools and functions.

IMPORTANT: When the user asks you to perform an action (like turning on lights, adding items to lists, controlling devices), you MUST call the appropriate tool. Do not just describe what you would do - actually call the tool.

Examples:
- "Add milk to shopping list" → Call HassListAddItem tool
- "Turn on the lights" → Call HassTurnOn tool
- "What's on my todo list?" → Call todo_get_items tool

Always prefer using tools over generating a text-only response when a tool is available.""")

            # Prepend system message (only if not already present)
            messages_with_system = [system_prompt] + messages

            # Bind tools to the device model
            llm_with_tools = self.chat_device.bind_tools(tools)

            self.logger.info(f"[agent] Invoking LLM with {len(tools)} tools")
            response = llm_with_tools.invoke(messages_with_system)

            # Check if LLM wants to call tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.logger.info(f"[agent] LLM requested {len(response.tool_calls)} tool calls")
                return {
                    "messages": [response]
                }
            else:
                # Log if we expected tool calls but didn't get them
                self.logger.warning(f"[agent] LLM did not call tools despite having {len(tools)} available")
                self.logger.debug(f"[agent] Response content: {response.content[:200] if response.content else 'None'}")

        # No tools or no tool calls - use regular LLM
        self.logger.info("[agent] Generating final response without tools")
        response = self.chat_device.invoke(messages)

        return {
            "messages": [response]
        }

    def _should_continue_to_tools(self, state: RouterState) -> str:
        """Determine if we need to execute tools or format response."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            self.logger.info("[should_continue_to_tools] Tool calls detected, returning for execution")
            return "tools"

        self.logger.info("[should_continue_to_tools] No tool calls, proceeding to formatter")
        return "formatter"

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
            state_update = {
                "messages": tool_messages
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
                "tools": tools
            }

            result = await self.graph.ainvoke(initial_state, config)

        # Check what the graph returned
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]

            # Check if we need tool execution
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                self.logger.info(f"[process] Returning {len(last_message.tool_calls)} tool calls")

                # Format tool calls for Home Assistant
                tool_calls = []
                for tc in last_message.tool_calls:
                    tool_calls.append({
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tc["name"],
                        "args": {k: v for k, v in tc["args"].items() if v is not None}
                    })

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
