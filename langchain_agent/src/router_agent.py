import logging
import os
import json

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from ollama import generate

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



        # self.router_llm = ChatOpenAI(
        #     temperature=0.0,
        #     name=ROUTER_MODEL,
        # )
        # self.device_llm = ChatOpenAI(
        #     temperature=0.0,
        #     name=DEVICE_MODEL,
        # )
        # self.query_llm = ChatOpenAI(
        #     temperature=0.7,
        #     model=QUERY_MODEL,
        # )

        # self.router_runnable = ROOT_CLASSIFIER_PROMPT | self.router_llm
        # self.device_runnable = DEVICE_PROMPT | self.device_llm
        # self.general_runnable = GENERAL_PROMPT | self.query_llm

    def route(self, query: str) -> str:
        self.logger.info("Routing query: %s", query)
        # output_text = self._invoke(self.root_classifier, {"query": query}).strip().lower()
        tool = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=5,
            topic="general",
            # include_answer=False,
            # include_raw_content=False,
            # include_images=False,
            # include_image_descriptions=False,
            # search_depth="basic",
            # time_range="day",
            # include_domains=None,
            # exclude_domains=None
        )
        model_generated_tool_call = {
            "args": {"query": query},
            "id": "1",
            "name": "tavily",
            "type": "tool_call",
        }
        search_response = tool.invoke(model_generated_tool_call)
        search_response_content = json.loads(search_response.content)
        search_results = search_response_content['results']
        if len(search_results) > 0:
            top_results = search_results[:5]
            output_text = self._invoke(self.search_results_combiner,{"query": query, "search_results": str(top_results)}).strip()
        else:
            output_text = "I don't know. Please ask me something else."

        self.logger.info("Routing decision: %s", output_text)

        return output_text
        # if route_outcome == "device":
        #     return self._invoke(self.device_runnable, {"command": query}).strip()
        # return self._invoke(self.general_runnable, {"question": query}).strip()

    @staticmethod
    def _invoke(runnable, args: dict) -> str:
        result = runnable.invoke(args)
        print(F"Result: {result}")
        return result['text']
        # if hasattr(result, "content"):
        #     return result.content
        # if isinstance(result, list) and result and hasattr(result[0], "content"):
        #     return result[0].content
        # return str(result)


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
