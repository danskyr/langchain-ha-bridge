"""Conversation platform for LangChain Remote."""
import logging

import aiohttp
from homeassistant.components import assist_pipeline, conversation as conversation
from homeassistant.components.conversation import AbstractConversationAgent, ConversationResult, ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponseErrorCode

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation platform."""
    async_add_entities([RemoteConversationAgent(hass, config_entry)])


class RemoteConversationAgent(AbstractConversationAgent, ConversationEntity):
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super.__init__()
        self.hass = hass
        self.entry = entry
        self._name = "LangChain Conversation Agent"

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return ["en"]

    async def _async_handle_message(
            self,
            user_input: conversation.ConversationInput,
            chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        # TODO: Use chat log to keep context and pass that along as well. Look into OpenAI Conversation here:
        #       https://github.com/home-assistant/core/blob/dev/homeassistant/components/openai_conversation/conversation.py

        entry_id = list(self.hass.data[DOMAIN].keys())[0]
        configuration = self.hass.data[DOMAIN][entry_id]
        url = configuration.get("url")
        timeout = configuration.get("timeout", 10)
        verify_ssl = configuration.get("verify_ssl", True)

        intent_response = intent.IntentResponse(language=user_input.language)
        try:
            session = async_get_clientsession(self.hass, verify_ssl=verify_ssl)

            response = await session.post(
                url,
                json={"text": user_input.text},
                timeout=aiohttp.ClientTimeout(total=timeout)
            )

            if response.status == 200:
                try:
                    response_body = await response.json()
                    response_text = response_body["response"]
                    intent_response.async_set_speech(response_text)
                except Exception:
                    _LOGGER.error("Failed to parse as JSON and extract response from LangChain service: %s", await response.text())
                    # intent_response.async_set_speech("Sorry, something went wrong.")
                    intent_response.async_set_error(IntentResponseErrorCode.FAILED_TO_HANDLE,
                                                    "Failed to parse response from LangChain service.")
            else:
                intent_response.async_set_error(IntentResponseErrorCode.FAILED_TO_HANDLE,
                                                f"Invalid response from LangChain service. Status code: {response.status}")

        except aiohttp.ClientConnectorError:
            intent_response.async_set_error(IntentResponseErrorCode.FAILED_TO_HANDLE,
                                            f"Connection to LangChain service refused.")
        except aiohttp.ClientSSLError:
            intent_response.async_set_error(IntentResponseErrorCode.FAILED_TO_HANDLE,
                                            f"SSL Error when connecting to LangChain service.")
        except Exception as err:
            _LOGGER.error("Unexpected error testing connection: %s", err)
            intent_response.async_set_error(IntentResponseErrorCode.FAILED_TO_HANDLE,
                                            f"Unknown error when connecting to LangChain service.")

        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
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
