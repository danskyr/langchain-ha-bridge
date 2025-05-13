import logging
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-3.5-turbo")
QUERY_MODEL = os.getenv("QUERY_MODEL", "gpt-4")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt templates
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
    """
    A router agent that classifies user queries and dispatches them
    to the appropriate LLMChain (device control vs general Q&A).
    """

    def __init__(self) -> None:
        # Validate environment
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")

        # Initialize LLMs
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

        # Initialize chains
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
        """
        Route the query to either the device or general chain.

        Args:
            query: The user input string.
        Returns:
            The output from the selected chain.
        """
        logger.info("Routing query: %s", query)
        decision = self.router_chain.run(query=query).strip().lower()
        logger.info("Routing decision: %s", decision)

        if decision == "device":
            return self.device_chain.run(command=query)
        return self.general_chain.run(question=query)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LangChain Router Agent: IoT device control vs general Q&A"
    )
    parser.add_argument(
        "query", help="The user query to classify and execute"
    )
    args = parser.parse_args()

    agent = LangChainRouterAgent()
    output = agent.route(args.query)
    print(output)
