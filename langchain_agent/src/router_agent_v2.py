import logging
import os
import uuid
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from .state import RouterState
from .utils import create_tavily_tool, preview_text
from .nodes import (
    router_node,
    route_to_handlers,
    iot_handler_node,
    general_handler_node,
    aggregator_node,
    create_agent_node,
    create_validation_node,
    validation_decision,
    formatter_node,
    announcement_node,
)

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

        self.local_tools = []
        tavily_tool = create_tavily_tool()
        if tavily_tool:
            self.local_tools.append(tavily_tool)
            self.logger.info(f"Initialized {len(self.local_tools)} local tool(s): Tavily search")
        else:
            self.logger.warning("Tavily API key not found - web search will not be available")

        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the state graph with checkpointing and interrupts."""
        workflow = StateGraph(RouterState)

        agent_node = create_agent_node(self)
        validation_node = create_validation_node(self)

        workflow.add_node("router", router_node)
        workflow.add_node("announcement", announcement_node)
        workflow.add_node("iot_handler", iot_handler_node)
        workflow.add_node("general_handler", general_handler_node)
        workflow.add_node("aggregator", aggregator_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tool_call_validation", validation_node)
        workflow.add_node("formatter", formatter_node)

        if self.local_tools:
            workflow.add_node("local_tools", ToolNode(self.local_tools))

        workflow.add_edge(START, "router")

        # After router, branch to both announcement and handlers
        workflow.add_conditional_edges(
            "router",
            route_to_handlers,
            ["iot_handler", "general_handler"]
        )

        # Also run announcement in parallel (non-blocking)
        workflow.add_edge("router", "announcement")

        # Handlers go to aggregator
        workflow.add_edge("iot_handler", "aggregator")
        workflow.add_edge("general_handler", "aggregator")

        # Announcement also goes to aggregator (to ensure it completes before agent)
        workflow.add_edge("announcement", "aggregator")

        workflow.add_edge("aggregator", "agent")

        workflow.add_edge("agent", "tool_call_validation")

        validation_edge_map = {
            "retry": "agent",
            "ha_tools": END,
            "formatter": "formatter",
        }

        if self.local_tools:
            validation_edge_map["local_tools"] = "local_tools"
            workflow.add_edge("local_tools", "agent")

        workflow.add_conditional_edges(
            "tool_call_validation",
            validation_decision,
            validation_edge_map
        )

        workflow.add_edge("formatter", END)

        return workflow.compile(checkpointer=self.checkpointer)

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
        thread_id = conversation_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        self.logger.info(f"[process] Thread: {thread_id[:8]}... | Tool results: {bool(tool_results)}")

        if tool_results:
            self.logger.info(f"[process] Resuming with {len(tool_results)} tool results")

            tool_messages = []
            for result in tool_results:
                tool_messages.append(
                    ToolMessage(
                        content=str(result.get("result", "")),
                        tool_call_id=result.get("tool_call_id", "")
                    )
                )

            state_update = {
                "messages": tool_messages,
                "validation_attempts": 1
            }

            result = await self.graph.ainvoke(state_update, config)
        else:
            self.logger.info(f"[process] Starting new conversation")

            initial_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "route_types": [],
                "handler_responses": [],
                "final_response": None,
                "tools": tools,
                "validation_attempts": 1,
                "preliminary_messages": [],
                "streaming_events": []
            }

            result = await self.graph.ainvoke(initial_state, config)

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                self.logger.info(f"[process] Returning {len(last_message.tool_calls)} tool calls")

                ha_tools = result.get("tools", [])
                tool_defs_by_name = {tool.get("function", {}).get("name"): tool for tool in ha_tools}

                tool_calls = []
                for i, tc in enumerate(last_message.tool_calls, 1):
                    formatted_tc = {
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tc["name"],
                        "args": {k: v for k, v in tc["args"].items() if v is not None}
                    }
                    tool_calls.append(formatted_tc)

                    self.logger.info(f"[process]   Tool call #{i}:")
                    self.logger.info(f"[process]     - Name: {formatted_tc['name']}")
                    self.logger.info(f"[process]     - ID: {formatted_tc['id']}")
                    self.logger.info(f"[process]     - Args: {formatted_tc['args']}")

                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "conversation_id": thread_id
                }

        final_response = result.get("final_response", "No response generated")
        continue_conversation = result.get("continue_conversation")
        self.logger.info(f"[process] Returning final response: {preview_text(final_response, 100)}")
        self.logger.info(f"[process] continue_conversation: {continue_conversation}")

        return {
            "type": "response",
            "response": final_response,
            "continue_conversation": continue_conversation
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
