import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

dotenv_result = load_dotenv()
if not dotenv_result:
    logging.warning(".env file not found or could not be loaded.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-3.5-turbo")
DEVICE_MODEL = os.getenv("DEVICE_MODEL", "gpt-3.5-turbo")
QUERY_MODEL = os.getenv("QUERY_MODEL", "gpt-4")

ROUTER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
You have two chains available:
- device: for Home Assistant / IoT service calls
- general: for general Q&A

Decide which chain should handle the user input.
Output exactly one word: device or general.

User input:
{query}
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


class LangChainRouterAgent:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")

        self.router_llm = ChatOpenAI(
            temperature=0.0,
            name=ROUTER_MODEL,
        )
        self.device_llm = ChatOpenAI(
            temperature=0.0,
            name=DEVICE_MODEL,
        )
        self.query_llm = ChatOpenAI(
            temperature=0.7,
            model=QUERY_MODEL,
        )

        self.router_runnable = ROUTER_PROMPT | self.router_llm
        self.device_runnable = DEVICE_PROMPT | self.device_llm
        self.general_runnable = GENERAL_PROMPT | self.query_llm

    def route(self, query: str) -> str:
        self.logger.info("Routing query: %s", query)
        route_outcome = self._invoke(self.router_runnable, {"query": query}).strip().lower()
        self.logger.info("Routing decision: %s", route_outcome)

        if route_outcome == "device":
            return self._invoke(self.device_runnable, {"command": query}).strip()
        return self._invoke(self.general_runnable, {"question": query}).strip()

    @staticmethod
    def _invoke(runnable, args: dict) -> str:
        result = runnable.invoke(args)
        if hasattr(result, "content"):
            return result.content
        if isinstance(result, list) and result and hasattr(result[0], "content"):
            return result[0].content
        return str(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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
