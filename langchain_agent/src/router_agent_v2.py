import logging
import os
import uuid
from dataclasses import asdict
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from .state import RouterState
from .utils import create_tavily_tool, preview_text
from .execution_tracer import ExecutionTracer, ExecutionTrace
from .langfuse_setup import create_langfuse_handler
from .nodes import (
    router_node,
    route_to_handlers,
    create_iot_handler_node,
    create_search_handler_node,
    create_general_handler_node,
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

MAX_HISTORY_MESSAGES = 10


def convert_ha_messages_to_langchain(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
    """Convert Home Assistant message format to LangChain BaseMessage types."""
    result: List[BaseMessage] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        elif role == "tool_result":
            tool_result_content = msg.get("tool_result", content)
            result.append(ToolMessage(
                content=str(tool_result_content),
                tool_call_id=msg.get("tool_call_id", ""),
                name=msg.get("tool_name", "unknown")
            ))

    return result


class LangChainRouterAgentV2:
    """
    Router agent with meaningful handler nodes.

    Graph flow:
    START -> router -> [iot_handler | search_handler | general_handler] + announcement
                    -> aggregator -> agent (tool-result only) -> tool_call_validation -> formatter -> END

    Each handler does the actual LLM work for its intent type.
    The agent node only handles tool-result continuation after HA executes tools.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger('langchain_agent.LangChainRouterAgentV2')

        # self.chat_router = ChatOllama(
        #     model="llama3.2:3b",
        #     base_url="http://localhost:11434",
        #     temperature=0
        # )
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

        self.local_tools: list = []
        tavily_tool = create_tavily_tool()
        if tavily_tool:
            self.local_tools.append(tavily_tool)
            self.logger.info(f"Initialized {len(self.local_tools)} local tool(s): Tavily search")
        else:
            self.logger.warning("Tavily API key not found - web search will not be available")

        self.checkpointer = MemorySaver()
        self.tracer = ExecutionTracer()
        self.graph = self._build_graph(checkpointer=self.checkpointer)

    def _build_graph(self, checkpointer=None) -> StateGraph:
        """Build the state graph with exclusive handler routing."""
        workflow = StateGraph(RouterState)

        agent_node = create_agent_node(self)
        validation_node = create_validation_node(self)
        iot_handler = create_iot_handler_node(self)
        search_handler = create_search_handler_node(self)
        general_handler = create_general_handler_node(self)

        workflow.add_node("router", router_node)
        workflow.add_node("announcement", announcement_node)
        workflow.add_node("iot_handler", iot_handler)
        workflow.add_node("search_handler", search_handler)
        workflow.add_node("general_handler", general_handler)
        workflow.add_node("aggregator", aggregator_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tool_call_validation", validation_node)
        workflow.add_node("formatter", formatter_node)

        if self.local_tools:
            workflow.add_node("local_tools", ToolNode(self.local_tools))

        workflow.add_edge(START, "router")

        # Exclusive routing: exactly one handler runs per query
        workflow.add_conditional_edges(
            "router",
            route_to_handlers,
            ["iot_handler", "search_handler", "general_handler"]
        )

        # Announcement runs in parallel with the selected handler
        workflow.add_edge("router", "announcement")

        # All handlers converge to aggregator
        workflow.add_edge("iot_handler", "agent")
        workflow.add_edge("search_handler", "agent")
        workflow.add_edge("general_handler", "agent")
        workflow.add_edge("announcement", "agent")

        # workflow.add_edge("aggregator", "agent")
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

        return workflow.compile(checkpointer=checkpointer)

    async def _invoke_with_tracing(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke graph with event streaming to capture execution trace."""
        result: Dict[str, Any] = {}

        graph_nodes = {"router", "announcement", "iot_handler", "search_handler",
                       "general_handler", "aggregator", "agent", "tool_call_validation",
                       "formatter", "local_tools"}

        async for event in self.graph.astream_events(state, config, version="v2"):
            event_type = event.get("event", "")
            event_name = event.get("name", "")

            if event_type == "on_chain_start" and event_name in graph_nodes:
                metadata = {}
                if event_name == "router":
                    data = event.get("data", {})
                    if "input" in data and isinstance(data["input"], dict):
                        metadata["route_types"] = data["input"].get("route_types", [])
                self.tracer.record_node_start(event_name, metadata)

            elif event_type == "on_chain_end" and event_name in graph_nodes:
                output = event.get("data", {}).get("output", {})
                if event_name == "router" and isinstance(output, dict):
                    for node in self.tracer.current_trace.nodes if self.tracer.current_trace else []:
                        if node.name == "router":
                            node.metadata["route_types"] = output.get("route_types", [])
                elif event_name == "agent" and isinstance(output, dict):
                    messages = output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for node in self.tracer.current_trace.nodes if self.tracer.current_trace else []:
                                if node.name == "agent":
                                    node.metadata["tool_calls"] = last_msg.tool_calls
                elif event_name in ("iot_handler", "search_handler", "general_handler") and isinstance(output, dict):
                    messages = output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for node in self.tracer.current_trace.nodes if self.tracer.current_trace else []:
                                if node.name == event_name:
                                    node.metadata["tool_calls"] = last_msg.tool_calls
                elif event_name == "tool_call_validation" and isinstance(output, dict):
                    for node in self.tracer.current_trace.nodes if self.tracer.current_trace else []:
                        if node.name == "tool_call_validation":
                            messages = output.get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                    node.metadata["decision"] = "ha_tools"
                                elif output.get("final_response"):
                                    node.metadata["decision"] = "formatter"

                self.tracer.record_node_end(event_name)

            elif event_type == "on_chain_end" and event_name == "LangGraph":
                result = event.get("data", {}).get("output", {})

        return result

    async def process(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process messages from HA with the graph.

        The messages array contains the full chat history including:
        - {"role": "system", "content": "..."}
        - {"role": "user", "content": "..."}
        - {"role": "assistant", "content": "...", "tool_calls": [...]}
        - {"role": "tool_result", "tool_call_id": "...", "tool_name": "...", "tool_result": {...}}
        """
        thread_id = conversation_id or str(uuid.uuid4())
        config: Dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "run_name": "ha_conversation",
            "metadata": {
                "langfuse_session_id": thread_id,
                "langfuse_user_id": "home_assistant",
                "langfuse_tags": ["ha_bridge"],
            },
        }
        langfuse_handler = create_langfuse_handler()
        if langfuse_handler is not None:
            config["callbacks"] = [langfuse_handler]

        self.tracer.start_trace(thread_id)
        self.logger.info(f"[process] Thread: {thread_id[:8]}... | Messages: {len(messages)}")

        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        if messages and messages[-1].get("role") == "tool_result":
            tool_results = []
            for m in reversed(messages):
                if m.get("role") == "tool_result":
                    tool_results.insert(0, {
                        "tool_call_id": m.get("tool_call_id"),
                        "tool_name": m.get("tool_name"),
                        "result": m.get("tool_result")
                    })
                else:
                    break
            self.logger.info(f"[process] Resuming with {len(tool_results)} tool results")
            for tr in tool_results:
                self.logger.info(f"[process]   - {tr.get('tool_name', 'unknown')}: {str(tr.get('result', ''))[:100]}")

            self.tracer.record_ha_roundtrip(tool_results)

            tool_messages = [
                ToolMessage(
                    content=str(tr.get("result", "")),
                    tool_call_id=tr.get("tool_call_id", ""),
                    name=tr.get("tool_name", "unknown")
                )
                for tr in tool_results
            ]

            state_update: Dict[str, Any] = {
                "messages": tool_messages,
                "validation_attempts": 1
            }

            result = await self._invoke_with_tracing(state_update, config)
        else:
            self.logger.info(f"[process] Starting new/continuing conversation")

            existing_checkpoint = self.checkpointer.get(config)
            has_existing_state = (
                existing_checkpoint is not None
                and isinstance(existing_checkpoint, dict)
                and existing_checkpoint.get("channel_values", {}).get("messages")
            )
            self.logger.info(f"[process] has_existing_state: {has_existing_state}")

            if has_existing_state:
                self.logger.info(f"[process] Thread has existing state, only adding new query")
                conversation_history: List[BaseMessage] = []
            else:
                history_messages: List[Dict[str, Any]] = []
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        history_messages = messages[:i]
                        break

                conversation_history = convert_ha_messages_to_langchain(history_messages)
                if len(conversation_history) > MAX_HISTORY_MESSAGES:
                    conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

            self.logger.info(f"[process] Including {len(conversation_history)} history messages for context")

            initial_state: Dict[str, Any] = {
                "messages": conversation_history + [HumanMessage(content=query)],
                "query": query,
                "route_types": [],
                "handler_responses": [],
                "final_response": None,
                "tools": tools,
                "validation_attempts": 1,
                "preliminary_messages": [],
                "active_handler": None
            }

            result = await self._invoke_with_tracing(initial_state, config)

        result_messages = result.get("messages", [])
        if result_messages:
            last_message = result_messages[-1]

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                self.logger.info(f"[process] Returning {len(last_message.tool_calls)} tool calls")

                self.tracer.end_trace(final=False)

                ha_tools = result.get("tools", [])
                tool_defs_by_name = {tool.get("function", {}).get("name"): tool for tool in ha_tools}

                tool_calls = []
                for i, tc in enumerate(last_message.tool_calls, 1):
                    formatted_tc = {
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tc["name"],
                        "args": {k: v for k, v in tc["args"].items() if v is not None and v != ""}
                    }
                    tool_calls.append(formatted_tc)

                    self.logger.info(f"[process]   Tool call #{i}:")
                    self.logger.info(f"[process]     - Name: {formatted_tc['name']}")
                    self.logger.info(f"[process]     - ID: {formatted_tc['id']}")
                    self.logger.info(f"[process]     - Args: {formatted_tc['args']}")

                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "conversation_id": thread_id,
                    "execution_trace": None
                }

        execution_trace = self.tracer.end_trace(final=True)

        final_response = result.get("final_response", "No response generated")
        continue_conversation = result.get("continue_conversation")
        self.logger.info(f"[process] Returning final response: {preview_text(final_response, 100)}")
        self.logger.info(f"[process] continue_conversation: {continue_conversation}")

        return {
            "type": "response",
            "response": final_response,
            "continue_conversation": continue_conversation,
            "execution_trace": asdict(execution_trace) if execution_trace else None
        }


if __name__ == "__main__":
    import asyncio

    agent = LangChainRouterAgentV2()
    print("Welcome to the LangChain Router Agent V2!")
    print("Type your query, or 'exit' to quit.")

    async def run_cli():
        try:
            while True:
                input_query = input("You: ")
                if not input_query.strip():
                    continue
                if input_query.strip().lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                messages = [{"role": "user", "content": input_query}]
                response = await agent.process(messages=messages)
                print(f"Agent: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")

    asyncio.run(run_cli())
