import logging
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-3.5-turbo")
QUERY_MODEL = os.getenv("QUERY_MODEL", "gpt-4")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
You are a helpful assistant. Answer the following question in clear, concise language.
Question: {question}
""",
)


class LangChainRouterAgent:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")

        self.router_llm = ChatOpenAI(
            temperature=0.0,
            model_name=ROUTER_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        self.query_llm = ChatOpenAI(
            temperature=0.7,
            model_name=QUERY_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )

        self.router_chain: LLMChain = LLMChain(
            llm=self.router_llm,
            prompt=ROUTER_PROMPT,
        )
        self.device_chain: LLMChain = LLMChain(
            llm=self.router_llm,
            prompt=DEVICE_PROMPT,
        )
        self.general_chain: LLMChain = LLMChain(
            llm=self.query_llm,
            prompt=GENERAL_PROMPT,
        )

    def route(self, query: str) -> str:
        logger.info("Routing query: %s", query)
        decision = self.router_chain.run(query=query).strip().lower()
        logger.info("Routing decision: %s", decision)

        if decision == "device":
            return self.device_chain.run(command=query)
        return self.general_chain.run(question=query)


if __name__ == "__main__":
    agent = LangChainRouterAgent()
    print("Welcome to the LangChain Router Agent! 👋")
    print("Type your query, or 'exit' to quit.")
    try:
        while True:
            query = input("You: ")
            if not query:
                continue
            if query.strip().lower() in ("exit", "quit"):
                print("Goodbye! 👋")
                break
            response = agent.route(query)
            print(f"Agent: {response}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye! 👋")
