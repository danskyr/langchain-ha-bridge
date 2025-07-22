import logging
import os
from typing import TypedDict, Literal

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

ROOT_CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent routing agent for a voice assistant. Classify the user's request into one of the following categories: 'IOT_COMMAND' (any simple command to turn on/off an IoT device) or 'ADVANCED_REQUEST' (everything else). Provide only the category.

Request: "{query}"
""",
)

SEARCH_RESULTS_COMBINER_PROMPT = PromptTemplate(
    input_variables=["query", "search_results"],
    template="""
Review the user's query and relevant search results and craft a short answer to the user's query. Reply only with this answer.

User's Query: 
```
{query}
```

Relevant Search Results:
```
{search_results}
```
""",
)

DEVICE_PROMPT = PromptTemplate(
    input_variables=["command"],
    template="""
You are an IoT controller. Interpret the user command and output a Home Assistant service call payload in YAML.
User Command: {command}
Output only the YAML service call.
""",
)

GENERAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful voice assistant. Answer the following question in clear, concise language.
Question: {question}
""",
)


class AgentState(TypedDict):
    query: str
    classification: str
    search_results: str
    final_response: str


from langchain_community.llms import Ollama

gemma2b = Ollama(base_url="http://localhost:11434", model="gemma:2b-instruct-q2_K")
mistral7b = Ollama(base_url="http://localhost:11434", model="mistral:7b")


class LangChainRouterAgent:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")

        self.root_classifier = LLMChain(llm=gemma2b, prompt=ROOT_CLASSIFIER_PROMPT)
        self.search_results_combiner = LLMChain(llm=gemma2b, prompt=SEARCH_RESULTS_COMBINER_PROMPT)

        self.tavily_search = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=5,
            topic="general"
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("classify", self._classify_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("respond", self._respond_node)

        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges(
            "classify",
            self._should_search,
            {
                "search": "search",
                "respond": "respond"
            }
        )
        workflow.add_edge("search", "respond")
        workflow.add_edge("respond", END)

        return workflow.compile()

    def _classify_node(self, state: AgentState) -> AgentState:
        self.logger.info("Classifying query: %s", state["query"])

        result = self.root_classifier.invoke({"query": state["query"]})
        classification = result['text'].strip().lower()

        self.logger.info("Classification result: %s", classification)

        return {
            **state,
            "classification": classification
        }

    def _should_search(self, state: AgentState) -> Literal["search", "respond"]:
        classification = state["classification"]

        if "iot_command" in classification:
            return "respond"

        return "search"

    def _search_node(self, state: AgentState) -> AgentState:
        self.logger.info("Searching for: %s", state["query"])

        try:
            search_results = self.tavily_search.invoke({"query": state["query"]})
            search_results_str = str(search_results)

            self.logger.info("Search completed, found results")

            return {
                **state,
                "search_results": search_results_str
            }
        except Exception as e:
            self.logger.error("Search failed: %s", e)
            return {
                **state,
                "search_results": "No search results available"
            }

    def _respond_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        classification = state["classification"]
        search_results = state.get("search_results", "")

        self.logger.info("Generating response for classification: %s", classification)

        if "iot_command" in classification:
            response = f"IoT command detected: {query}. This would be processed by the device controller."
        elif search_results and search_results != "No search results available":
            result = self.search_results_combiner.invoke({
                "query": query,
                "search_results": search_results
            })
            response = result['text'].strip()
        else:
            response = "I don't have enough information to answer that question. Please try asking something else."

        self.logger.info("Generated response: %s", response)

        return {
            **state,
            "final_response": response
        }

    def route(self, query: str) -> str:
        self.logger.info("Routing query: %s", query)

        initial_state = AgentState(
            query=query,
            classification="",
            search_results="",
            final_response=""
        )

        final_state = self.graph.invoke(initial_state)

        response = final_state["final_response"]
        self.logger.info("Final response: %s", response)

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
