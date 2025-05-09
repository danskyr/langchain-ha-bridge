"""Conversation platform for LangChain Remote."""
from homeassistant.components.conversation import ConversationAgent, ConversationInput
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import DiscoveryInfoType

from .const import DOMAIN
from .ha_proxy import RemoteConversationAgent

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: DiscoveryInfoType,
) -> None:
    """Set up the conversation platform."""
    agent = RemoteConversationAgent(hass)
    async_add_entities([agent])
