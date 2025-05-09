# custom_components/langchain_remote/config_flow.py
import voluptuous as vol
from homeassistant import config_entries
from .const import DOMAIN

DATA_SCHEMA = vol.Schema({
    vol.Required("url", default="http://127.0.0.1:8000/process"): str
})

class LangChainRemoteConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=DATA_SCHEMA)
        return self.async_create_entry(title="LangChain Remote", data=user_input)