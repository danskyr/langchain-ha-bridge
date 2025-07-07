import voluptuous as vol
import aiohttp
import asyncio
from homeassistant import config_entries
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.core import callback
from .const import DOMAIN
import logging

from .utils import get_host_from_url

_LOGGER = logging.getLogger(__name__)

DATA_SCHEMA = vol.Schema({
    vol.Required("url", default="http://127.0.0.1:8000", description="LangChain Service URL"): str,
    vol.Optional("timeout", default=10, description="Connection timeout (seconds)"): vol.All(vol.Coerce(int), vol.Range(min=1, max=60)),
    vol.Optional("verify_ssl", default=False, description="Verify SSL certificates"): bool,
})

class LangChainRemoteConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self.data = {}
        self.errors = {}

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            # Store the configuration
            self.data = user_input

            # Validate URL format
            url_validation = self._validate_url_format(user_input.get("url"))
            if not url_validation["valid"]:
                errors["url"] = url_validation["error"]
            else:
                # Test connection to the URL
                connection_test = await self._test_connection(user_input)
                if connection_test["valid"]:
                    return await self.async_step_confirm()
                else:
                    errors["base"] = connection_test["error"]

        return self.async_show_form(
            step_id="user",
            data_schema=DATA_SCHEMA,
            errors=errors,
            description_placeholders={
                "url_example": "Example: http://192.168.1.100:8000",
                "timeout_help": "How long to wait for responses (1-60 seconds)",
            },
        )

    async def async_step_confirm(self, user_input=None):
        """Confirm the configuration."""
        if user_input is not None or self.data:
            return self.async_create_entry(
                title=f"LangChain Conversation Agent API ({get_host_from_url(self.data['url'])})",
                description="LangChain Conversation Agent API",
                data=self.data
            )

        return self.async_show_form(
            step_id="confirm",
            description_placeholders={
                "url": self.data.get("url"),
                "timeout": self.data.get("timeout", 10),
                "verify_ssl": "Yes" if self.data.get("verify_ssl", False) else "No",
            },
        )

    def _validate_url_format(self, url):
        """Validate URL format."""
        if not url:
            return {"valid": False, "error": "url_required"}

        if not url.startswith(("http://", "https://")):
            return {"valid": False, "error": "url_invalid_protocol"}

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.netloc:
                return {"valid": False, "error": "url_invalid_format"}
        except Exception:
            return {"valid": False, "error": "url_invalid_format"}

        return {"valid": True, "error": None}

    async def _test_connection(self, config):
        """Test connection to the LangChain service."""
        url = config.get("url")
        timeout = config.get("timeout", 10)
        verify_ssl = config.get("verify_ssl", False)

        try:
            session = async_get_clientsession(self.hass, verify_ssl=verify_ssl)

            # Test with a simple health check or ping
            test_payload = {
                "message": "Connection test",
                "test": True
            }

            async with asyncio.timeout(timeout):
                async with session.post(
                        url + "/test",
                        json=test_payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        try:
                            await response.json()
                            return {"valid": True, "error": None}
                        except Exception:
                            return {"valid": False, "error": f"received 200, but failed to parse response as json: {await response.text()}"}
                    else:
                        return {"valid": False, "error": f"connection_error_status_{response.status}"}

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
        """Get the options flow for this handler."""
        return LangChainRemoteOptionsFlowHandler(config_entry)


class LangChainRemoteOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for LangChain Remote integration."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(
                    "timeout",
                    default=self.config_entry.options.get("timeout", 10)
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=60)),
                vol.Optional(
                    "verify_ssl",
                    default=self.config_entry.options.get("verify_ssl", False)
                ): bool,
            })
        )