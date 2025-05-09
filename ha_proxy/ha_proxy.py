import requests
from homeassistant.components.conversation import AbstractConversationAgent
from homeassistant.core import HomeAssistant
from .const import DOMAIN

class RemoteConversationAgent(AbstractConversationAgent):
    def __init__(self, hass: HomeAssistant):
        super().__init__(hass)
        # self._url = hass.data[DOMAIN].config.get("url")

    async def async_process(self, text: str, conversation_id=None) -> str:
        # fire off HTTP POST to your LangChain server
        resp = requests.post("http://192.168.1.66:8001/process", json={"text": text}, timeout=10)
        resp.raise_for_status()
        return resp.json()["response"]