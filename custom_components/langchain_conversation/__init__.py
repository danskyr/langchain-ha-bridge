"""The LangChain Remote integration."""
from __future__ import annotations

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN
from .client import LangChainClient, WebSocketLogHandler

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

    url = entry.data.get("url")
    verify_ssl = entry.data.get("verify_ssl", False)

    client = LangChainClient(hass, url, verify_ssl)

    # Connect with timeout - ConfigEntryNotReady triggers HA's retry mechanism
    try:
        async with asyncio.timeout(CONNECT_TIMEOUT):
            if not await client.connect():
                raise ConfigEntryNotReady("Failed to connect to LangChain server")
    except TimeoutError:
        raise ConfigEntryNotReady("Connection to LangChain server timed out")
    except Exception as err:
        _LOGGER.error("Error connecting to LangChain server: %s", err)
        raise ConfigEntryNotReady(f"Connection error: {err}") from err

    # Set up log forwarding to server
    log_handler = WebSocketLogHandler(client)
    log_handler.setFormatter(logging.Formatter('%(message)s'))
    log_handler.setLevel(logging.DEBUG)

    # Add handler to component logger (and all child loggers)
    # __name__ is the component path, e.g., custom_components.langchain_conversation
    component_logger = logging.getLogger(__name__)
    component_logger.addHandler(log_handler)

    # Start the log forwarding task
    await log_handler.start()
    _LOGGER.info("Log forwarding to LangChain server enabled")

    # Cleanup on HA shutdown
    async def _shutdown(event):
        await log_handler.stop()
        await client.disconnect()

    entry.async_on_unload(
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _shutdown)
    )

    # Ensure cleanup on unload
    async def _cleanup():
        component_logger.removeHandler(log_handler)
        await log_handler.stop()
        await client.disconnect()

    entry.async_on_unload(_cleanup)

    # Store client and config data
    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
        "log_handler": log_handler,
        **entry.data
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        # Client disconnect is handled by async_on_unload callback

    return unload_ok
