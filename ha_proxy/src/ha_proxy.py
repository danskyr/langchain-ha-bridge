import requests
from homeassistant.components.conversation import AbstractConversationAgent
from homeassistant.core import HomeAssistant

DATA_SCHEMA = vol.Schema({
    vol.Required("url", default="http://127.0.0.1:8000/process"): str
})


class RemoteConversationAgent(AbstractConversationAgent):
    def __init__(self, hass: HomeAssistant):
        super().__init__(hass)
        self._url = hass.data[DOMAIN].config.get("url")

    async def async_process(self, text: str, conversation_id=None) -> str:
        # fire off HTTP POST to your LangChain server
        resp = requests.post(self._url, json={"text": text}, timeout=10)
        resp.raise_for_status()
        return resp.json()["response"]