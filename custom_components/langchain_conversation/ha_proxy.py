import requests
from homeassistant.components import assist_pipeline, conversation as conversation
from homeassistant.components.conversation import AbstractConversationAgent, ConversationResult, ConversationEntity
from homeassistant.core import HomeAssistant

from .const import DOMAIN


class RemoteConversationAgent(AbstractConversationAgent, ConversationEntity):
    def __init__(self, hass: HomeAssistant):
        AbstractConversationAgent.__init__(self)
        ConversationEntity.__init__(self, hass)
        self.hass = hass
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

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()
