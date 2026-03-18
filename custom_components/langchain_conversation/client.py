"""HTTP client for OpenClaw server."""
import json
import logging
from typing import Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)


class OpenClawClient:
    """HTTP client for OpenClaw server communication."""

    def __init__(
        self,
        hass: HomeAssistant,
        url: str,
        api_key: str,
        verify_ssl: bool = True,
        agent_id: str = "main",
        session_key: Optional[str] = None,
    ) -> None:
        self.hass = hass
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._agent_id = agent_id
        self._session_key = session_key

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "x-openclaw-agent-id": self._agent_id,
        }
        if self._session_key:
            headers["x-openclaw-session-key"] = self._session_key
        return headers

    async def check_connection(self) -> bool:
        """Check connectivity via GET /v1/models."""
        session = async_get_clientsession(self.hass, verify_ssl=self._verify_ssl)
        try:
            async with session.get(
                f"{self._url}/v1/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            ) as resp:
                return resp.status < 500
        except Exception as err:
            _LOGGER.debug("OpenClaw connection check failed: %s", err)
            return False

    async def chat(self, messages: list[dict]) -> str:
        """Send messages to OpenClaw and return the full response text."""
        session = async_get_clientsession(self.hass, verify_ssl=self._verify_ssl)
        payload = {"messages": messages, "stream": True}

        response_text = ""
        async with session.post(
            f"{self._url}/v1/chat/completions",
            headers=self._build_headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    response_text += content
                except json.JSONDecodeError:
                    _LOGGER.debug("Failed to parse SSE chunk: %s", data)

        return response_text
