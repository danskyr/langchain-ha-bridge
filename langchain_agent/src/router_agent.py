import os

from dotenv import load_dotenv
# from langchain.chains.router import RouterChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
router_model = os.environ.get("ROUTER_MODEL", "gpt-3.5-turbo")
query_model = os.environ.get("QUERY_MODEL", "gpt-4")


class LangChainRouterAgent:
    """Router agent for LangChain.
    
    This class is a placeholder for the actual implementation.
    """

    def __init__(self):
        """Initialize the router agent."""
        pass

    def route(self, query):
        # Get configuration from environment variables

        # Init LLMs
        router_llm = ChatOpenAI(
            temperature=0.0,
            model_name=router_model,
            openai_api_key=openai_api_key
        )

        query_llm = ChatOpenAI(
            temperature=0.7,
            model_name=query_model,
            openai_api_key=openai_api_key
        )

        device_chain = LLMChain(
            llm=router_llm,
            prompt=PromptTemplate(
                input_variables=["command"],
                template="""
        You are an IoT controller. Interpret the user command and output a Home Assistant service call payload in YAML.
        User Command: {command}
        Output only the YAML service call.
        """,
            ),
        )

        # Define a general query chain
        general_chain = LLMChain(
            llm=query_llm,
            prompt=PromptTemplate(
                input_variables=["question"],
                template="""
        You are a helpful assistant. Answer the following question in clear, concise language.
        Question: {question}
        """,
            ),
        )

        # Setup a simple router (RouterChain API has changed)
        # For now, we'll just use a simple function to route requests
        # In a real implementation, this would use the router_llm to determine the route
        if "turn on" in query.lower() or "turn off" in query.lower():
            return device_chain.run(command=query)
        else:
            return general_chain.run(question=query)
