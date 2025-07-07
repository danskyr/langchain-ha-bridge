import logging
import os

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
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

ROOT_CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent routing agent for a voice assistant. Classify the user's request into one of the following categories: 'IOT_COMMAND' (any simple command to turn on/off an IoT device) or 'ADVANCED_REQUEST' (everything else). Provide only the category.

Request: "{query}"
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

gemma2b = Ollama(base_url="http://127.0.0.1:11434", model="gemma:2b-instruct-q2_K")

class LangChainRouterAgent:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")

        self.root_classifier = LLMChain(llm=gemma2b, prompt=ROOT_CLASSIFIER_PROMPT)



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
        route_outcome = self._invoke(self.root_classifier, {"query": query}).strip().lower()
        self.logger.info("Routing decision: %s", route_outcome)

        return "Thanks!"
        # if route_outcome == "device":
        #     return self._invoke(self.device_runnable, {"command": query}).strip()
        # return self._invoke(self.general_runnable, {"question": query}).strip()

    @staticmethod
    def _invoke(runnable, args: dict) -> str:
        result = runnable.invoke(args)
        if hasattr(result, "content"):
            return result.content
        if isinstance(result, list) and result and hasattr(result[0], "content"):
            return result[0].content
        return str(result)


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
