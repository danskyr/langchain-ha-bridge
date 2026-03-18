import asyncio
import logging

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN, CONF_API_KEY, CONF_AGENT_ID, CONF_SESSION_KEY
from .utils import get_host_from_url

_LOGGER = logging.getLogger(__name__)

DATA_SCHEMA = vol.Schema({
    vol.Required("url", default="http://host.docker.internal:18791"): str,
    vol.Required(CONF_API_KEY): str,
    vol.Optional(CONF_AGENT_ID, default="main"): str,
    vol.Optional(CONF_SESSION_KEY, default=""): str,
    vol.Optional("timeout", default=90): vol.All(vol.Coerce(int), vol.Range(min=1, max=180)),
    vol.Optional("verify_ssl", default=True): bool,
})


class LangChainRemoteConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    def __init__(self) -> None:
        self.data: dict = {}
        self.errors: dict = {}

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors: dict = {}

        if user_input is not None:
            self.data = user_input

            url_validation = self._validate_url_format(user_input.get("url"))
            if not url_validation["valid"]:
                errors["url"] = url_validation["error"]
            else:
                connection_test = await self._test_connection(user_input)
                if connection_test["valid"]:
                    return await self.async_step_confirm()
                else:
                    errors["base"] = connection_test["error"]

        return self.async_show_form(
            step_id="user",
            data_schema=DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_confirm(self, user_input=None):
        """Confirm the configuration."""
        if user_input is not None or self.data:
            url = self.data.get("url", "")
            return self.async_create_entry(
                title=f"OpenClaw ({get_host_from_url(url) if url else 'unknown'})",
                data=self.data,
            )

        return self.async_show_form(
            step_id="confirm",
            description_placeholders={
                "url": self.data.get("url"),
                "agent_id": self.data.get(CONF_AGENT_ID, "main"),
            },
        )

    def _validate_url_format(self, url: str | None) -> dict:
        if not url:
            return {"valid": False, "error": "url_required"}
        if not url.startswith(("http://", "https://")):
            return {"valid": False, "error": "url_invalid_protocol"}
        try:
            from urllib.parse import urlparse
            if not urlparse(url).netloc:
                return {"valid": False, "error": "url_invalid_format"}
        except Exception:
            return {"valid": False, "error": "url_invalid_format"}
        return {"valid": True, "error": None}

    async def _test_connection(self, config: dict) -> dict:
        """Test connection via GET /v1/models."""
        url = config.get("url", "").rstrip("/")
        api_key = config.get(CONF_API_KEY, "")
        verify_ssl = config.get("verify_ssl", True)
        timeout_val = config.get("timeout", 10)

        try:
            session = async_get_clientsession(self.hass, verify_ssl=verify_ssl)
            async with asyncio.timeout(timeout_val):
                async with session.get(
                    f"{url}/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                ) as resp:
                    if resp.status == 401:
                        return {"valid": False, "error": "invalid_api_key"}
                    if resp.status >= 500:
                        return {"valid": False, "error": "connection_unknown_error"}
                    return {"valid": True, "error": None}

        except asyncio.TimeoutError:
            return {"valid": False, "error": "connection_timeout"}
        except aiohttp.ClientConnectorError:
            return {"valid": False, "error": "connection_refused"}
        except aiohttp.ClientSSLError:
            return {"valid": False, "error": "ssl_error"}
        except Exception as err:
            _LOGGER.error("Unexpected error testing connection: %s", err)
            return {"valid": False, "error": "connection_unknown_error"}

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return LangChainRemoteOptionsFlowHandler(config_entry)


class LangChainRemoteOptionsFlowHandler(config_entries.OptionsFlow):

    def __init__(self, config_entry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(
                    "timeout",
                    default=self.config_entry.options.get("timeout", 90),
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=180)),
                vol.Optional(
                    "verify_ssl",
                    default=self.config_entry.options.get("verify_ssl", True),
                ): bool,
            }),
        )
