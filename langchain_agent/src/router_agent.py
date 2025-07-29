import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypedDict, Optional, Dict, Any, List

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

dotenv_result = load_dotenv()
if not dotenv_result:
    logging.warning(".env file not found or could not be loaded.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-3.5-turbo")
DEVICE_MODEL = os.getenv("DEVICE_MODEL", "gpt-3.5-turbo")
QUERY_MODEL = os.getenv("QUERY_MODEL", "gpt-4")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "gpt-4")


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
    handler_data: Dict[
        str, Any]  # todo might make it hard to use with context, since the overview of what type of data / source is gone. Maybe the key should be another enum. Or we need a dataclass
    raw_response: str
    final_response: str
    response_type: ResponseType


class BaseHandler(ABC):
    """Abstract base class for all query handlers."""

    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)

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


class ResponseFormatter:
    """Unified response formatter for consistent tone and format."""

    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

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

            self.logger.info(f"Raw response:\n{state["raw_response"]}")

            chain = LLMChain(llm=self.llm, prompt=self.format_prompt)
            result = chain.invoke({
                "query": state["query"],
                "response": state["raw_response"],
                "response_type": state["response_type"].value
            })

            formatted_response = result['text'].strip()

            return {
                **state,
                "final_response": formatted_response
            }

        except Exception as e:
            self.logger.error("Failed to format response: %s", e)
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
        try:
            result = LLMChain(llm=self.llm, prompt=self.classifier_prompt).invoke({"query": query})
            classification = result['text'].strip().upper()
            return "YES" in classification
        except Exception as e:
            self.logger.error("Error in IOT classification: %s", e)
            return False

    def get_route_type(self) -> RouteType:
        return RouteType.IOT_COMMAND

    def process(self, state: AgentState) -> AgentState:
        try:
            result = LLMChain(llm=self.llm, prompt=self.device_prompt).invoke({"command": state["query"]})
            response = result['text'].strip()

            return {
                **state,
                "raw_response": f"I'll help you control your device. {response}",
                "response_type": ResponseType.SUCCESS,
                "handler_data": {"yaml_output": response}
            }
        except Exception as e:
            self.logger.error("Error processing IoT command: %s", e)
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

            search_results_str = ""
            for result in search_results:
                search_results_str += (
                    f"# {result['title']}\n"
                    f"## Relevance score\n{result['score']}\n"
                    f"## Content\n{result['content']}\n---\n\n"
                )

            result = LLMChain(llm=self.llm, prompt=self.combiner_prompt).invoke({
                "query": state["query"],
                "search_results": search_results_str
            })

            response = result['text'].strip()

            return {
                **state,
                "raw_response": response,
                "response_type": ResponseType.INFO,
                "handler_data": {"search_results": search_results_str}
            }
        except Exception as e:
            self.logger.error("Error in search processing: %s", e)
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
            result = LLMChain(llm=self.llm, prompt=self.general_prompt).invoke({"question": state["query"]})
            response = result['text'].strip()

            return {
                **state,
                "raw_response": response,
                "response_type": ResponseType.INFO,
                "handler_data": {}
            }
        except Exception as e:
            self.logger.error("Error in general processing: %s", e)
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
        self.logger = logging.getLogger(__name__)

        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")

        self.tavily_search = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=5,
            topic="general"
        )

        self.handlers: List[BaseHandler] = [
            IOTHandler(gemma2b),
            SearchHandler(qwen3b, self.tavily_search),
            GeneralHandler(gemma2b)
        ]

        self.response_formatter = ResponseFormatter(qwen3b)
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
        self.logger.info("Routing query: %s", query)

        for handler in self.handlers:  # todo should run them concurrently
            if handler.can_handle(query):
                route_type = handler.get_route_type()
                self.logger.info("Query routed to: %s via %s", route_type.value, handler.__class__.__name__)

                return {  # todo multiple handlers should actually be able to run
                    **state,
                    "route_type": route_type,
                    "handler_data": {"selected_handler": handler}
                }

        self.logger.warning("No handler found for query, defaulting to GeneralHandler")
        return {
            **state,
            "route_type": RouteType.GENERAL_QUERY,
            "handler_data": {"selected_handler": self.handlers[-1]}
        }

    def _process_node(self, state: AgentState) -> AgentState:
        """Process the query using the selected handler."""
        handler = state["handler_data"]["selected_handler"]
        self.logger.info("Processing with handler: %s", handler.__class__.__name__)

        return handler.process(state)

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response for consistent user experience."""
        self.logger.info("Formatting response for type: %s", state.get("response_type", "unknown"))

        return self.response_formatter.format_response(state)

    def route(self, query: str) -> str:
        self.logger.info("Processing query: %s", query)

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
        self.logger.info(f"Final response:\n{response}")

        return response


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
