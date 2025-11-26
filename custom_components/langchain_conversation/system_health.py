"""System health for LangChain Remote integration."""
from __future__ import annotations

from typing import Any

from homeassistant.components.system_health import SystemHealthRegistration
from homeassistant.core import HomeAssistant, callback

from .const import DOMAIN


@callback
def async_register(
    hass: HomeAssistant, register: SystemHealthRegistration
) -> None:
    """Register system health callbacks."""
    register.async_register_info(system_health_info)


async def system_health_info(hass: HomeAssistant) -> dict[str, Any]:
    """Get info for the info page."""
    if not hass.config_entries.async_entries(DOMAIN):
        return {"status": "not_configured"}

    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    url = config_entry.data.get("url", "unknown")

    # Get WebSocket connection status from stored client
    data = hass.data.get(DOMAIN, {}).get(config_entry.entry_id, {})
    client = data.get("client")

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    return {
        "api_endpoint": url,
        "websocket_url": ws_url,
        "websocket_connected": client.is_connected if client else False,
    }
