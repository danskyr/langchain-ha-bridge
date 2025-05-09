"""LangChain Router Agent for Home Assistant."""
import logging
import voluptuous as vol
from typing import Any, Dict, Optional

from homeassistant.components.conversation import AbstractConversationAgent, ConversationResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import TemplateVarsType

from langchain.chat_models import ChatOpenAI
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain import LLMChain
from langchain.prompts import PromptTemplate

_LOGGER = logging.getLogger(__name__)

# Configuration schema
CONFIG_SCHEMA = vol.Schema({
    vol.Required("openai_api_key"): cv.string,
    vol.Optional("router_model", default="gpt-3.5-turbo"): cv.string,
    vol.Optional("query_model", default="gpt-4"): cv.string,
})

class LangChainRouterAgent(AbstractConversationAgent):
    """LangChain Router Agent for Home Assistant."""

    def __init__(self, hass: HomeAssistant, config: Dict[str, Any]):
        """Initialize the agent."""
        super().__init__(hass)
        self.hass = hass
        self.config = config
        self.openai_api_key = config.get("openai_api_key")
        self.router_model = config.get("router_model", "gpt-3.5-turbo")
        self.query_model = config.get("query_model", "gpt-4")
        
        # Initialize LLMs
        self._initialize_llms()
        
        # Initialize chains
        self._initialize_chains()

    def _initialize_llms(self):
        """Initialize the language models."""
        try:
            # LLM for routing (deterministic classification)
            self.router_llm = ChatOpenAI(
                temperature=0.0,
                model_name=self.router_model,
                openai_api_key=self.openai_api_key
            )
            
            # LLM for handling general queries (higher capability)
            self.query_llm = ChatOpenAI(
                temperature=0.7,
                model_name=self.query_model,
                openai_api_key=self.openai_api_key
            )
        except Exception as e:
            _LOGGER.error("Error initializing LLMs: %s", e)
            raise

    def _initialize_chains(self):
        """Initialize the LangChain chains."""
        try:
            # Define the device-control chain
            device_prompt = PromptTemplate(
                input_variables=["command"],
                template="""
You are an IoT controller for Home Assistant. Interpret the user command and output a Home Assistant service call payload in YAML.
Examples of commands:
- "Turn on the living room lights" -> service: light.turn_on, entity_id: light.living_room
- "Set the thermostat to 72 degrees" -> service: climate.set_temperature, entity_id: climate.thermostat, temperature: 72

User Command: {command}
Output only the YAML service call.
"""
            )
            self.device_chain = LLMChain(llm=self.router_llm, prompt=device_prompt)

            # Define a general query chain
            query_prompt = PromptTemplate(
                input_variables=["question"],
                template="""
You are a helpful assistant integrated with Home Assistant. Answer the following question in clear, concise language.
If the question is about home automation or smart home concepts, provide informative responses.
If asked about specific device status, explain that you don't have real-time access to device states.

Question: {question}
"""
            )
            self.general_chain = LLMChain(llm=self.query_llm, prompt=query_prompt)

            # Define router prompt
            router_prompt = PromptTemplate(
                template=MULTI_PROMPT_ROUTER_TEMPLATE,
                input_variables=["input", "destinations"]
            )

            # Define destination chains with descriptions
            destinations = {
                "device": "Commands to control smart home devices, turn things on/off, adjust settings, etc.",
                "query": "General questions, information requests, or conversations not related to controlling devices"
            }
            
            destination_chains = {
                "device": self.device_chain,
                "query": self.general_chain
            }

            # Create router chain
            router_chain = LLMRouterChain.from_llm(
                llm=self.router_llm,
                prompt=router_prompt,
                destinations=destinations,
                output_parser=RouterOutputParser()
            )

            # Create the multi-prompt chain
            self.chain = MultiPromptChain(
                router_chain=router_chain,
                destination_chains=destination_chains,
                default_chain=self.general_chain,
                verbose=True
            )
        except Exception as e:
            _LOGGER.error("Error initializing chains: %s", e)
            raise

    async def async_process(self, text: str, conversation_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> ConversationResult:
        """Process a user query."""
        try:
            # Run the router synchronously in executor
            result = await self.hass.async_add_executor_job(
                lambda: self.chain.run(text)
            )
            
            return ConversationResult(
                response=result,
                conversation_id=conversation_id
            )
        except Exception as e:
            _LOGGER.error("Error processing query: %s", e)
            return ConversationResult(
                response=f"I'm sorry, I encountered an error: {str(e)}",
                conversation_id=conversation_id
            )

async def async_setup(hass: HomeAssistant, config: Dict[str, Any]) -> bool:
    """Set up the LangChain Router Agent."""
    conf = config.get("langchain_agent", {})
    
    try:
        # Validate config
        conf = CONFIG_SCHEMA(conf)
        
        # Register the conversation agent
        hass.data.setdefault("conversation", {})["langchain_agent"] = LangChainRouterAgent(hass, conf)
        
        return True
    except Exception as e:
        _LOGGER.error("Error setting up LangChain Router Agent: %s", e)
        return False