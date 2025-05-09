import requests
from homeassistant.components.conversation import AbstractConversationAgent, ConversationResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from .const import DOMAIN

class RemoteConversationAgent(AbstractConversationAgent):
    def __init__(self, hass: HomeAssistant):
        super().__init__(hass)
        self._hass = hass
        self._name = "LangChain Remote Agent"

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return ["en"]

    @property
    def attribution(self) -> str:
        """Return the attribution."""
        return "Powered by LangChain"

    async def async_process(self, text: str, conversation_id=None, context=None) -> ConversationResult:
        """Process a sentence."""
        entry_id = list(self._hass.data[DOMAIN].keys())[0]
        url = self._hass.data[DOMAIN][entry_id].get("url", "http://127.0.0.1:8000/process")

        # fire off HTTP POST to your LangChain server
        resp = requests.post(url, json={"text": text}, timeout=10)
        resp.raise_for_status()
        response_text = resp.json()["response"]

        return ConversationResult(
            response=response_text,
            conversation_id=conversation_id,
        )
