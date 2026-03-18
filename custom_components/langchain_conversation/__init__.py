"""The LangChain Remote integration."""
from __future__ import annotations

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, CONF_API_KEY, CONF_AGENT_ID, CONF_SESSION_KEY
from .client import OpenClawClient

_LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION,)
CONNECT_TIMEOUT = 10


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the LangChain Remote component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up LangChain Remote from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    client = OpenClawClient(
        hass,
        url=entry.data["url"],
        api_key=entry.data[CONF_API_KEY],
        verify_ssl=entry.data.get("verify_ssl", True),
        agent_id=entry.data.get(CONF_AGENT_ID, "main"),
        session_key=entry.data.get(CONF_SESSION_KEY) or None,
    )

    try:
        async with asyncio.timeout(CONNECT_TIMEOUT):
            if not await client.check_connection():
                raise ConfigEntryNotReady("Failed to connect to OpenClaw server")
    except TimeoutError:
        raise ConfigEntryNotReady("Connection to OpenClaw server timed out")
    except ConfigEntryNotReady:
        raise
    except Exception as err:
        _LOGGER.error("Error connecting to OpenClaw server: %s", err)
        raise ConfigEntryNotReady(f"Connection error: {err}") from err

    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
        **entry.data,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
