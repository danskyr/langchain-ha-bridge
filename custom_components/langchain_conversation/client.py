"""WebSocket client for LangChain server."""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)


class LangChainClient:
    """WebSocket client for LangChain server communication."""

    def __init__(self, hass: HomeAssistant, url: str, verify_ssl: bool = False):
        self.hass = hass
        self._url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._verify_ssl = verify_ssl
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            session = async_get_clientsession(self.hass, verify_ssl=self._verify_ssl)
            self._ws = await session.ws_connect(self._url, heartbeat=30)
            self._connected = True
            _LOGGER.info("WebSocket connected to LangChain server at %s", self._url)
            return True
        except aiohttp.ClientError as e:
            _LOGGER.error("WebSocket connection failed: %s", e)
            self._connected = False
            return False
        except Exception as e:
            _LOGGER.error("Unexpected error connecting to WebSocket: %s", e)
            self._connected = False
            return False

    async def disconnect(self):
        """Close WebSocket connection."""
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            _LOGGER.info("WebSocket disconnected from LangChain server")

    async def send_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send conversation and wait for response.

        Args:
            conversation_id: Unique identifier for the conversation
            messages: Full chat history including tool calls and results
            tools: Available HA tools

        Returns:
            Response dict with type, content, and optionally tool_calls
        """
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")

        await self._ws.send_json({
            "type": "conversation",
            "conversation_id": conversation_id,
            "messages": messages,
            "tools": tools
        })

        # Wait for response matching our conversation_id
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                # Only return if this response is for our conversation
                if data.get("conversation_id") == conversation_id:
                    return data
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                _LOGGER.warning("WebSocket closed while waiting for response")
                raise ConnectionError("WebSocket closed unexpectedly")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                _LOGGER.error("WebSocket error: %s", self._ws.exception())
                raise ConnectionError(f"WebSocket error: {self._ws.exception()}")

        raise ConnectionError("No response received")

    async def send_log(self, level: str, message: str):
        """Forward log entry to server (best effort)."""
        if self.is_connected:
            try:
                await self._ws.send_json({
                    "type": "log",
                    "level": level,
                    "message": message
                })
            except Exception:
                pass  # Best effort - don't fail if log forwarding fails

    async def ping(self) -> bool:
        """Send ping and wait for pong to verify connection."""
        if not self.is_connected:
            return False

        try:
            await self._ws.send_json({"type": "ping"})
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "pong":
                        return True
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    return False
        except Exception:
            return False
        return False
