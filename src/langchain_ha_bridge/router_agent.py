"""LangChain Router Agent implementation for Home Assistant."""
from typing import Dict, Any

from homeassistant.components.conversation import AbstractConversationAgent
from homeassistant.core import HomeAssistant
from langchain.chat_models import ChatOpenAI
from langchain.chains.router import RouterChain
from langchain import LLMChain
from langchain.prompts import PromptTemplate


class LangChainRouterAgent(AbstractConversationAgent):
    """LangChain Router Agent for Home Assistant.
    
    This agent uses LangChain to route user queries to the appropriate handler
    based on intent, either controlling smart home devices or answering general questions.
    """
    
    def __init__(self, hass: HomeAssistant, config: Dict[str, Any] = None):
        """Initialize the agent.
        
        Args:
            hass: Home Assistant instance
            config: Configuration dictionary with optional keys:
                - openai_api_key: OpenAI API key
                - router_model: Model to use for routing (default: gpt-3.5-turbo)
                - query_model: Model to use for general queries (default: gpt-4)
        """
        super().__init__(hass)
        
        # Get configuration or use defaults
        config = config or {}
        openai_api_key = config.get("openai_api_key", "YOUR_OPENAI_API_KEY")
        router_model = config.get("router_model", "gpt-3.5-turbo")
        query_model = config.get("query_model", "gpt-4")
        
        # LLM for routing (deterministic classification)
        self.router_llm = ChatOpenAI(
            temperature=0.0,
            model_name=router_model,
            openai_api_key=openai_api_key
        )
        
        # LLM for handling general queries (higher capability)
        self.query_llm = ChatOpenAI(
            temperature=0.7,
            model_name=query_model,
            openai_api_key=openai_api_key
        )

        # Define the device-control chain
        device_prompt = PromptTemplate(
            input_variables=["command"],
            template="""
You are an IoT controller. Interpret the user command and output a Home Assistant service call payload in YAML.
User Command: {command}
Output only the YAML service call.
"""
        )
        self.device_chain = LLMChain(llm=self.router_llm, prompt=device_prompt)

        # Define a general query chain
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
You are a helpful assistant. Answer the following question in clear, concise language.
Question: {question}
"""
        )
        self.general_chain = LLMChain(llm=self.query_llm, prompt=query_prompt)

        # Setup RouterChain to classify into 'device' or 'query'
        chains = {
            "device": self.device_chain,
            "query": self.general_chain
        }
        default_chain = self.general_chain
        self.router = RouterChain.from_llm(
            llm=self.router_llm,
            chains=chains,
            default_chain=default_chain
        )

    async def async_process(self, text: str) -> str:
        """Process a user query.
        
        Args:
            text: The user's query text
            
        Returns:
            The response from the appropriate chain
        """
        # Run the router synchronously in executor
        result = await self.hass.async_add_executor_job(lambda: self.router.run(text))
        return result