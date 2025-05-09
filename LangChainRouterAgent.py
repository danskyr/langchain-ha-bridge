# Directory structure:
# custom_components/langchain_agent/
# ├─ __init__.py
# ├─ manifest.json
# └─ conversation.py

# __init__.py (empty file)

# manifest.json
# ----------------
# {
#   "domain": "langchain_agent",
#   "name": "LangChain Router Agent",
#   "version": "1.0.0",
#   "integration_type": "conversation_agent",
#   "requirements": ["langchain>=0.0.XXX", "openai"],
#   "dependencies": []
# }

# conversation.py
# ----------------
from homeassistant.components.conversation import AbstractConversationAgent
from homeassistant.core import HomeAssistant
from langchain.chat_models import ChatOpenAI
from langchain.chains.router import RouterChain
from langchain import LLMChain
from langchain.prompts import PromptTemplate

class LangChainRouterAgent(AbstractConversationAgent):
    def __init__(self, hass: HomeAssistant):
        super().__init__(hass)
        # LLM for routing (deterministic classification)
        self.router_llm = ChatOpenAI(
            temperature=0.0,
            model_name="gpt-3.5-turbo",
            openai_api_key="YOUR_OPENAI_API_KEY"
        )
        # LLM for handling general queries (higher capability)
        self.query_llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
            openai_api_key="YOUR_OPENAI_API_KEY"
        )

        # Define the device-control chain (you could replace this with direct service calls)
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
        # Run the router synchronously in executor
        result = await self.hass.async_add_executor_job(lambda: self.router.run(text))
        return result

# To enable this integration, place the folder under custom_components/, restart HA,
# then in configuration.yaml or via UI add:
# conversation:
#   integration: langchain_agent
