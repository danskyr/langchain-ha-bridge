"""System health for LangChain Remote integration."""
from __future__ import annotations

from typing import Any

from homeassistant.components.system_health import SystemHealthRegistration
from homeassistant.core import HomeAssistant, callback

from .const import DOMAIN, CONF_AGENT_ID


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
    data = config_entry.data

    return {
        "api_endpoint": data.get("url", "unknown"),
        "agent_id": data.get(CONF_AGENT_ID, "main"),
    }
