"""Conversation platform for LangChain Remote (OpenClaw backend)."""
import asyncio
import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AbstractConversationAgent,
    ConversationEntity,
    ConversationResult,
    async_get_chat_log,
    AssistantContent,
)
from homeassistant.components.conversation.util import async_get_result_from_chat_log
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers import device_registry as dr

from .const import DOMAIN
from .client import OpenClawClient

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation platform."""
    async_add_entities([RemoteConversationAgent(hass, config_entry)])


class RemoteConversationAgent(AbstractConversationAgent, ConversationEntity):
    _attr_supports_streaming = False

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super(AbstractConversationAgent, self).__init__()
        super(ConversationEntity, self).__init__()
        self.hass = hass
        self.entry = entry
        self._name = "LangChain Conversation Agent"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=self._name,
            default_name="LangChain Conversation Agent",
            manufacturer="LangChain",
            model="OpenClaw",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str]:
        return ["en"]

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> ConversationResult:
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        config = self.hass.data[DOMAIN][self.entry.entry_id]
        client: OpenClawClient = config["client"]
        timeout: int = config.get("timeout", 90)

        messages = self._chat_log_to_messages(chat_log)
        response_text = ""

        try:
            async with asyncio.timeout(timeout):
                response_text = await client.chat(messages)
        except TimeoutError:
            response_text = "Request timed out"
        except Exception as err:
            _LOGGER.error("Error communicating with OpenClaw: %s", err, exc_info=True)
            response_text = "An error occurred"

        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(agent_id=self.entity_id, content=response_text)
        )
        return async_get_result_from_chat_log(user_input, chat_log)

    def _chat_log_to_messages(
        self, chat_log: conversation.ChatLog
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for content in chat_log.content:
            if content.role in ("system", "user", "assistant"):
                messages.append({"role": content.role, "content": content.content or ""})
        return messages

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()
