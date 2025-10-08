import logging
import os
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypedDict, Optional, Dict, Any, List

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

logging.basicConfig(
    level=logging.WARNING,  # Keep root logger at WARNING to reduce library noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

dotenv_result = load_dotenv()
if not dotenv_result:
    logging.warning(".env file not found or could not be loaded.")

# Set up module-level logger with INFO level
module_logger = logging.getLogger('langchain_agent')
log_level = os.getenv('LANGCHAIN_LOG_LEVEL', 'INFO')
module_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-3.5-turbo")
DEVICE_MODEL = os.getenv("DEVICE_MODEL", "gpt-3.5-turbo")
QUERY_MODEL = os.getenv("QUERY_MODEL", "gpt-4")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "gpt-4")


def preview_text(text: str, max_len: int = 600) -> str:
    """Truncate text for logging, showing start and end for long content."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    half = (max_len - 5) // 2
    return f"{text[:half]} ... {text[-half:]}"


class RouteType(Enum):
    IOT_COMMAND = "iot_command"
    SEARCH_QUERY = "search_query"
    GENERAL_QUERY = "general_query"


class ResponseType(Enum):
    SUCCESS = "success"
    ERROR = "error"
    INFO = "info"


class AgentState(TypedDict):
    query: str
    route_type: Optional[RouteType]
    handler_data: Dict[str, Any]
    raw_response: str
    final_response: str
    response_type: ResponseType
    tools: Optional[List[Dict[str, Any]]]
    tool_calls: Optional[List[Dict[str, Any]]]
    tool_results: Optional[List[Dict[str, Any]]]
    conversation_id: Optional[str]
    messages: Optional[List[Dict[str, Any]]]
    needs_tool_execution: bool


class BaseHandler(ABC):
    """Abstract base class for all query handlers."""

    def __init__(self, llm):
        self.llm = llm
        self.llm_with_tools = None
        self.logger = logging.getLogger(f'langchain_agent.{self.__class__.__name__}')

    def bind_tools(self, tools: Optional[List[Dict[str, Any]]]):
        """Bind tools to the LLM for this handler."""
        if tools:
            self.llm_with_tools = self.llm.bind_tools(tools)
            self.logger.debug(f"Bound {len(tools)} tools to {self.__class__.__name__}")
        else:
            self.llm_with_tools = self.llm

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Determine if this handler can process the given query."""
        pass

    @abstractmethod
    def get_route_type(self) -> RouteType:
        """Return the route type this handler manages."""
        pass

    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """Process the query and update the state with results."""
        pass

    def process_with_tools(self, state: AgentState) -> AgentState:
        # state['raw_response'] = '{"name": "HassGet", "parameters": {"domain": "light", "attributes": "state"}}'
        # return state

        """Process query with tool support."""
        self.logger.debug(f"[process_with_tools] Processing query: {state['query'][:50]}...")  # Log first 50 chars only
        if not self.llm_with_tools:
            self.logger.debug(f"[process_with_tools] No tools bound for {self.__class__.__name__}")
            return self.process(state)

        # Create message for LLM
        messages = [HumanMessage(content=state["query"])]

        # If we have previous messages, include them
        if state.get("messages"):
            # Convert dict messages back to LangChain message objects
            for msg in state["messages"]:
                if msg["type"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "ai":
                    messages.append(AIMessage(content=msg["content"], tool_calls=msg.get("tool_calls", [])))
                elif msg["type"] == "tool":
                    messages.append(ToolMessage(content=msg["content"], tool_call_id=msg["tool_call_id"]))

        # Add tool results if any
        if state.get("tool_results"):
            for result in state["tool_results"]:
                messages.append(ToolMessage(
                    content=str(result.get("result", "")),
                    tool_call_id=result.get("tool_call_id", "")
                ))

        try:
            # Get model name if available
            model_name = getattr(self.llm_with_tools, 'model', getattr(self.llm_with_tools, 'model_name', 'unknown'))

            # Log input
            self.logger.info(f"  🤖 [process_with_tools] → {model_name}")
            for i, msg in enumerate(messages):
                msg_type = msg.__class__.__name__
                content_preview = preview_text(str(msg.content), 450)
                self.logger.info(f"     Msg {i+1} ({msg_type}): {content_preview}")

            # Get LLM response
            response = self.llm_with_tools.invoke(messages)

            # Log response
            has_tools = bool(getattr(response, 'tool_calls', None))
            if has_tools:
                self.logger.info(f"  ← [process_with_tools] Response: Tool call(s) requested")
            else:
                response_preview = preview_text(response.content if response.content else "", 450)
                self.logger.info(f"  ← [process_with_tools] Response: {response_preview}")

            # Convert messages to serializable format for state
            state["messages"] = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    state["messages"].append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    state["messages"].append({
                        "type": "ai",
                        "content": msg.content,
                        "tool_calls": getattr(msg, 'tool_calls', [])
                    })
                elif isinstance(msg, ToolMessage):
                    state["messages"].append({
                        "type": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id
                    })

            if hasattr(response, 'tool_calls') and response.tool_calls:
                # LLM wants to call tools
                tool_calls = []
                for tc in response.tool_calls:
                    tool_calls.append({
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tc["name"],
                        # Filter out None values - HA doesn't accept None for optional parameters
                        "args": {k: v for k, v in tc["args"].items() if v is not None}
                    })

                state["tool_calls"] = tool_calls
                state["needs_tool_execution"] = True
                state["raw_response"] = response.content or ""
                self.logger.info(f"  🔧 [process_with_tools] Tool calls requested:")
                for tc in tool_calls:
                    args_str = ', '.join(f"{k}={v}" for k, v in list(tc['args'].items())[:3])  # Show first 3 args
                    if len(tc['args']) > 3:
                        args_str += ", ..."
                    self.logger.info(f"     - {tc['name']}({args_str})")
            else:
                # Final response
                state["raw_response"] = response.content
                state["needs_tool_execution"] = False
                self.logger.info(f"  ✓ [process_with_tools] LLM provided final response")

        except Exception as e:
            self.logger.error(f"[process_with_tools] Error: {e}")
            state["raw_response"] = f"Error processing request: {str(e)}"
            state["needs_tool_execution"] = False

        return state


class ResponseFormatter:
    """Unified response formatter for consistent tone and format."""

    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger('langchain_agent.ResponseFormatter')

        self.format_prompt = PromptTemplate(
            input_variables=["response", "query"],
            template="""Consider these examples:
Examples:
---
Q: what's the weather?
Raw: The weather in Oklahoma City is very hot, with temperatures between 73°F and 93°F.
Short: It's very hot today, between 73 and 93 degrees.
---
Q: When did WW2 end?
Raw: [long explanation about the war ending in 1945]
Short: World War Two officially ended on September 2nd 1945.
---

Q: {query}
Raw: {response}

Given the Q (query) and Raw results rephrase as a short voice assistant response:
Short:"""
        )

    def format_response(self, state: AgentState) -> AgentState:
        """Format the raw response into a consistent, user-friendly format."""
        try:
            if not state.get("raw_response"):
                return {
                    **state,
                    "final_response": "I apologize, but I couldn't process your request. Please try asking again.",
                    "response_type": ResponseType.ERROR
                }

            model_name = getattr(self.llm, 'model', getattr(self.llm, 'model_name', 'unknown'))

            self.logger.info(f"  🤖 [format_response] → {model_name}")
            self.logger.info(f"     Raw: {preview_text(state['raw_response'], 450)}")

            chain = LLMChain(llm=self.llm, prompt=self.format_prompt)
            result = chain.invoke({
                "query": state["query"],
                "response": state["raw_response"],
                "response_type": state["response_type"].value
            })

            formatted_response = result['text'].strip()
            self.logger.info(f"  ← [format_response] Formatted: {preview_text(formatted_response, 450)}")

            return {
                **state,
                "final_response": formatted_response
            }

        except Exception as e:
            self.logger.error("[format_response] Failed to format: %s", e)
            return {
                **state,
                "final_response": state.get("raw_response", "I encountered an error processing your request."),
                "response_type": ResponseType.ERROR
            }


from langchain_community.llms import Ollama


class IOTHandler(BaseHandler):
    """Handler for IoT device commands."""

    def __init__(self, llm):
        super().__init__(llm)
        self.classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
Determine if this is a simple IoT device command (turn on/off, set temperature, etc.).
Respond with only 'YES' or 'NO'.

Query: "{query}"
""",
        )

        self.device_prompt = PromptTemplate(
            input_variables=["command"],
            template="""
You are an IoT controller. Interpret the user command and output a Home Assistant service call payload in YAML.
User Command: {command}
Output only the YAML service call.
""",
        )


    def can_handle(self, query: str) -> bool:
        return True

    # def can_handle(self, query: str) -> bool:
    #     try:
    #         result = LLMChain(llm=self.llm, prompt=self.classifier_prompt).invoke({"query": query})
    #         classification = result['text'].strip().upper()
    #         return "YES" in classification
    #     except Exception as e:
    #         self.logger.error("Error in IOT classification: %s", e)
    #         return False

    def get_route_type(self) -> RouteType:
        return RouteType.IOT_COMMAND

    def process(self, state: AgentState) -> AgentState:
        try:
            model_name = getattr(self.llm, 'model', getattr(self.llm, 'model_name', 'unknown'))

            self.logger.info(f"  🤖 [IOTHandler.process] → {model_name}")
            self.logger.info(f"     Query: {preview_text(state['query'], 450)}")

            result = LLMChain(llm=self.llm, prompt=self.device_prompt).invoke({"command": state["query"]})
            response = result['text'].strip()

            self.logger.info(f"  ← [IOTHandler.process] Response: {preview_text(response, 450)}")

            return {
                **state,
                "raw_response": f"I'll help you control your device. {response}",
                "response_type": ResponseType.SUCCESS,
                "handler_data": {"yaml_output": response}
            }
        except Exception as e:
            self.logger.error("[IOTHandler.process] Error: %s", e)
            return {
                **state,
                "raw_response": "I couldn't process that device command. Please try rephrasing your request.",
                "response_type": ResponseType.ERROR,
                "handler_data": {"error": str(e)}
            }


class SearchHandler(BaseHandler):
    """Handler for queries that require web search."""

    def __init__(self, llm, tavily_search):
        super().__init__(llm)
        self.tavily_search = tavily_search
        self.combiner_prompt = PromptTemplate(
            input_variables=["query", "search_results"],
            template="""You are an expert research analyst tasked with synthesizing search results to answer user queries accurately and comprehensively.

## Search Results
{search_results}

## User Query
{query}

## Instructions
Analyze the search results and provide a comprehensive response that:
- Directly addresses the user's specific question
- Synthesizes information from multiple sources when relevant
- Maintains factual accuracy and cites key sources
- Identifies any conflicting information or gaps in the data
- Uses clear, organized formatting appropriate to the query type

## Response
Provide your analysis below:""",
        )

    def can_handle(self, query: str) -> bool:
        return True

    def get_route_type(self) -> RouteType:
        return RouteType.SEARCH_QUERY

    def process(self, state: AgentState) -> AgentState:
        try:
            self.logger.info(f"  🔍 [SearchHandler.process] Searching for: '{preview_text(state['query'], 450)}'")

            tavily_response = self.tavily_search.invoke({"query": state["query"]})
            all_results = tavily_response.get("results", [])

            search_results = []
            if all_results:
                top_score = max(result.get('score', 0) for result in all_results)
                min_score_threshold = top_score * 0.9

                search_results = [
                    result for result in all_results
                    if result.get('score', 0) >= min_score_threshold
                ]

            self.logger.info(f"  ← [SearchHandler.process] Found {len(search_results)} relevant results")

            search_results_str = ""
            for result in search_results:
                search_results_str += (
                    f"# {result['title']}\n"
                    f"## Relevance score\n{result['score']}\n"
                    f"## Content\n{result['content']}\n---\n\n"
                )

            model_name = getattr(self.llm, 'model', getattr(self.llm, 'model_name', 'unknown'))
            self.logger.info(f"  🤖 [SearchHandler.process] → {model_name} (synthesizing from {len(search_results_str)} chars)")

            result = LLMChain(llm=self.llm, prompt=self.combiner_prompt).invoke({
                "query": state["query"],
                "search_results": search_results_str
            })

            response = result['text'].strip()
            self.logger.info(f"  ← [SearchHandler.process] Response: {preview_text(response, 450)}")

            return {
                **state,
                "raw_response": response,
                "response_type": ResponseType.INFO,
                "handler_data": {"search_results": search_results_str}
            }
        except Exception as e:
            self.logger.error("[SearchHandler.process] Error: %s", e)
            return {
                **state,
                "raw_response": "I couldn't find information about that topic. Please try asking something else.",
                "response_type": ResponseType.ERROR,
                "handler_data": {"error": str(e)}
            }


class GeneralHandler(BaseHandler):
    """Handler for general queries that don't require search."""

    def __init__(self, llm):
        super().__init__(llm)
        self.general_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
You are a helpful voice assistant. Answer the following question in clear, concise language.
Question: {question}
""",
        )

    def can_handle(self, query: str) -> bool:
        return True

    def get_route_type(self) -> RouteType:
        return RouteType.GENERAL_QUERY

    def process(self, state: AgentState) -> AgentState:
        try:
            model_name = getattr(self.llm, 'model', getattr(self.llm, 'model_name', 'unknown'))

            self.logger.info(f"  🤖 [GeneralHandler.process] → {model_name}")
            self.logger.info(f"     Query: {preview_text(state['query'], 450)}")

            result = LLMChain(llm=self.llm, prompt=self.general_prompt).invoke({"question": state["query"]})
            response = result['text'].strip()

            self.logger.info(f"  ← [GeneralHandler.process] Response: {preview_text(response, 450)}")

            return {
                **state,
                "raw_response": response,
                "response_type": ResponseType.INFO,
                "handler_data": {}
            }
        except Exception as e:
            self.logger.error("[GeneralHandler.process] Error: %s", e)
            return {
                **state,
                "raw_response": "I don't have enough information to answer that question. Please try asking something else.",
                "response_type": ResponseType.ERROR,
                "handler_data": {"error": str(e)}
            }


gemma2b = Ollama(base_url="http://localhost:11434", model="gemma:2b-instruct-q2_K")
mistral7b = Ollama(base_url="http://localhost:11434", model="mistral:7b")
phi4 = Ollama(base_url="http://localhost:11434", model="phi4:latest")
qwen3b = Ollama(base_url="http://localhost:11434", model="qwen2.5:3b")


class LangChainRouterAgent:
    def __init__(self) -> None:
        self.logger = logging.getLogger('langchain_agent.LangChainRouterAgent')

        # Initialize ChatOllama models for tool support
        # Using your available models - qwen2.5:7b and mistral:7b have best tool support
        self.chat_router = ChatOllama(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0
        )
        self.chat_device = ChatOllama(
            model="mistral:7b",  # Mistral for device control
            base_url="http://localhost:11434",
            temperature=0
        )
        self.chat_query = ChatOllama(
            model="qwen2.5:7b",  # Qwen for general queries
            base_url="http://localhost:11434",
            temperature=0
        )

        # Tavily search is optional - comment out if not needed
        if TAVILY_API_KEY:
            self.tavily_search = TavilySearch(
                tavily_api_key=TAVILY_API_KEY,
                max_results=5,
                topic="general"
            )
        else:
            self.tavily_search = None

        # Use ChatOllama models for handlers to support tools
        self.handlers: List[BaseHandler] = [
            IOTHandler(self.chat_router),
            # SearchHandler(self.chat_router, self.tavily_search) if self.tavily_search else None,
            # GeneralHandler(self.chat_query)
        ]
        # Remove None handler if search is disabled
        self.handlers = [h for h in self.handlers if h is not None]

        self.response_formatter = ResponseFormatter(self.chat_router)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # todo please advice on the best practice of this idea: each handler should be a node. and the handler nodes should be able to run in parallel. Then the final node that all handler nodes culminate in should be the final response node
        workflow.add_node("route", self._route_node)
        workflow.add_node("process", self._process_node)
        workflow.add_node("format_response", self._format_response_node)

        workflow.add_edge(START, "route")
        workflow.add_edge("route", "process")
        workflow.add_edge("process", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile()

    def _route_node(self, state: AgentState) -> AgentState:
        """Determine which handler should process this query."""
        query = state["query"]
        self.logger.info("[_route_node] → Routing query: %s", query[:100] + "..." if len(query) > 100 else query)

        for handler in self.handlers:  # todo should run them concurrently
            if handler.can_handle(query):
                route_type = handler.get_route_type()
                self.logger.info("[_route_node]   ↳ Routed to: %s", handler.__class__.__name__)

                return {  # todo multiple handlers should actually be able to run
                    **state,
                    "route_type": route_type,
                    "handler_data": {"selected_handler": handler}
                }

        self.logger.warning("[_route_node]   ⚠ No specific handler found, using GeneralHandler")
        return {
            **state,
            "route_type": RouteType.GENERAL_QUERY,
            "handler_data": {"selected_handler": self.handlers[-1]}
        }

    def _process_node(self, state: AgentState) -> AgentState:
        """Process the query using the selected handler."""
        handler = state["handler_data"]["selected_handler"]
        self.logger.info("[_process_node]   → Processing with: %s", handler.__class__.__name__)

        result_state = handler.process(state)

        # Log state transformation
        self.logger.debug(f"[_process_node]   State: raw_response={len(result_state.get('raw_response', ''))} chars, response_type={result_state.get('response_type')}")

        return result_state

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response for consistent user experience."""
        self.logger.debug("[_format_response_node] Formatting response")

        return self.response_formatter.format_response(state)

    def route(self, query: str) -> str:
        self.logger.info(f"[route] 📥 Query: '{preview_text(query, 900)}'")

        initial_state = AgentState(
            query=query,
            route_type=None,
            handler_data={},
            raw_response="",
            final_response="",
            response_type=ResponseType.INFO
        )

        final_state = self.graph.invoke(initial_state)

        response = final_state["final_response"]
        self.logger.info(f"[route] ✅ Final response: '{preview_text(response, 600)}'")

        return response

    async def route_with_tools(self, query: str, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Route query with tool support."""
        self.logger.info(f"[route_with_tools] 📥 Query: '{preview_text(query, 900)}'")
        self.logger.info(f"[route_with_tools]    Tools available: {len(tools) if tools else 0}")

        conversation_id = str(uuid.uuid4())

        # Create initial state
        initial_state = AgentState(
            query=query,
            route_type=None,
            handler_data={},
            raw_response="",
            final_response="",
            response_type=ResponseType.INFO,
            tools=tools,
            tool_calls=None,
            tool_results=None,
            conversation_id=conversation_id,
            messages=None,
            needs_tool_execution=False
        )

        # Route to appropriate handler and bind tools
        handler = None
        for h in self.handlers:
            if h.can_handle(query):
                handler = h
                break

        if not handler:
            return {
                "type": "response",
                "response": "I couldn't understand your request. Please try again."
            }

        # Bind tools to handler
        handler.bind_tools(tools)

        # Process with tools
        processed_state = handler.process_with_tools(initial_state)

        if processed_state.get("needs_tool_execution"):
            # Return tool calls for execution
            tool_count = len(processed_state["tool_calls"])
            self.logger.info(f"[route_with_tools] 🔧 Returning {tool_count} tool call(s) for execution")
            return {
                "type": "tool_call",
                "tool_calls": processed_state["tool_calls"],
                "conversation_id": conversation_id
            }
        else:
            # Format and return final response
            # formatted_state = self.response_formatter.format_response(processed_state)
            final_resp = processed_state["raw_response"]
            self.logger.info(f"[route_with_tools] ✅ Final response: '{preview_text(final_resp, 600)}'")
            return {
                "type": "response",
                "response": final_resp
            }

    async def route_with_tool_results(
        self,
        query: str,
        tools: Optional[List[Dict[str, Any]]],
        tool_results: List[Dict[str, Any]],
        conversation_id: Optional[str]
    ) -> str:
        """Continue conversation with tool results."""
        self.logger.info(f"[route_with_tool_results] 🔧 Continuing conversation {conversation_id[:8] if conversation_id else 'N/A'}...")
        self.logger.info(f"[route_with_tool_results]    Tool results: {len(tool_results)} result(s)")
        for tr in tool_results:
            result_preview = preview_text(str(tr.get('result', '')), 450)
            self.logger.info(f"[route_with_tool_results]      - {tr.get('name', 'unknown')}: {result_preview}")

        # Create state with tool results
        state = AgentState(
            query=query,
            route_type=None,
            handler_data={},
            raw_response="",
            final_response="",
            response_type=ResponseType.INFO,
            tools=tools,
            tool_calls=None,
            tool_results=tool_results,
            conversation_id=conversation_id,
            messages=None,
            needs_tool_execution=False
        )

        # Get the IOT handler (tool results likely come from IOT queries)
        handler = self.handlers[0]  # IOTHandler is first
        handler.bind_tools(tools)

        # Process with tool results
        self.logger.info(f"[route_with_tool_results]   → Processing {len(tool_results)} tool result(s)")
        processed_state = handler.process_with_tools(state)

        # Format final response
        # formatted_state = self.response_formatter.format_response(processed_state)
        final_resp = processed_state["raw_response"]
        self.logger.info(f"[route_with_tool_results] ✅ Final response: '{preview_text(final_resp, 600)}'")
        return final_resp


# Store conversation states (in production, use Redis or similar)
conversation_states = {}

if __name__ == "__main__":
    agent = LangChainRouterAgent()
    print("Welcome to the LangChain Router Agent! 👋")
    print("Type your query, or 'exit' to quit.")
    try:
        while True:
            input_query = input("You: ")
            if not input_query.strip():
                continue
            if input_query.strip().lower() in ("exit", "quit"):
                print("Goodbye! 👋")
                break
            response = agent.route(input_query)
            print(f"Agent: {response}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye! 👋")
