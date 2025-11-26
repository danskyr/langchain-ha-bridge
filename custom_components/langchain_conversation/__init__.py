"""The LangChain Remote integration."""
from __future__ import annotations

import asyncio
import logging

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN
from .client import LangChainClient

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

    # Background listener task to detect disconnects
    async def _listen_for_disconnect():
        """Monitor WebSocket connection and trigger reload on disconnect."""
        try:
            if client._ws is None:
                return

            async for msg in client._ws:
                if msg.type == aiohttp.WSMsgType.CLOSED:
                    _LOGGER.warning("WebSocket connection closed by server")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    _LOGGER.error("WebSocket error: %s", client._ws.exception())
                    break
                # Other message types are handled by send_conversation
        except Exception as e:
            _LOGGER.warning("WebSocket listener error: %s", e)
        finally:
            if not hass.is_stopping:
                _LOGGER.info("WebSocket disconnected, scheduling reload")
                hass.async_create_task(
                    hass.config_entries.async_reload(entry.entry_id)
                )

    listen_task = entry.async_create_background_task(
        hass, _listen_for_disconnect(), "langchain_ws_listen"
    )

    # Cleanup on HA shutdown
    async def _shutdown(event):
        await client.disconnect()

    entry.async_on_unload(
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _shutdown)
    )

    # Ensure client disconnects on unload
    async def _cleanup():
        await client.disconnect()

    entry.async_on_unload(_cleanup)

    # Store client and config data
    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
        "listen_task": listen_task,
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
