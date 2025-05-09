"""System health for LangChain Remote integration."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
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
    url = config_entry.data.get("url", "http://192.168.1.66:8001/process")

    data = {
        "api_endpoint": url,
    }

    # Add a connection test
    data["can_reach_server"] = await async_check_can_reach_server(hass, url)

    return data


async def async_check_can_reach_server(hass: HomeAssistant, url: str) -> str:
    """Test if we can reach the server."""
    try:
        # Use a simple test message
        test_data = {"text": "system health check"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=test_data, timeout=5) as resp:
                if resp.status == 200:
                    return "ok"
                return f"Error: {resp.status}"
    except (asyncio.TimeoutError, aiohttp.ClientError) as err:
        return f"Error: {err}"
