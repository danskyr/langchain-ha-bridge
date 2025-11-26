"""WebSocket client for LangChain server."""
import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, List

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)

# Reconnection settings
RECONNECT_MAX_DURATION = 600  # 10 minutes before giving up
RECONNECT_INITIAL_DELAY = 1  # Start with 1 second
RECONNECT_MAX_DELAY = 60  # Cap at 60 seconds
RECONNECT_BACKOFF_FACTOR = 2  # Double each time


class WebSocketLogHandler(logging.Handler):
    """Logging handler that forwards logs to LangChain server via WebSocket."""

    def __init__(self, client: "LangChainClient"):
        super().__init__()
        self.client = client
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def emit(self, record: logging.LogRecord):
        """Queue log record for async sending."""
        if self.client.is_connected:
            try:
                # Don't forward logs from this module to avoid loops
                if record.name.startswith("custom_components.langchain_conversation.client"):
                    return
                self._queue.put_nowait({
                    "level": record.levelname,
                    "message": self.format(record),
                    "logger": record.name,
                })
            except asyncio.QueueFull:
                pass  # Drop if queue is full

    async def start(self):
        """Start the background task that sends queued logs."""
        self._task = asyncio.create_task(self._send_logs())

    async def stop(self):
        """Stop the background task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _send_logs(self):
        """Send queued logs to server."""
        while True:
            try:
                log_entry = await self._queue.get()
                if self.client.is_connected:
                    await self.client.send_log(
                        log_entry["level"],
                        f"[{log_entry['logger']}] {log_entry['message']}"
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Best effort


class LangChainClient:
    """WebSocket client for LangChain server communication."""

    def __init__(self, hass: HomeAssistant, url: str, verify_ssl: bool = False):
        self.hass = hass
        self._url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._verify_ssl = verify_ssl
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._connected = False
        self._reconnect_start: Optional[float] = None
        self._reconnect_delay = RECONNECT_INITIAL_DELAY
        self._reconnect_logged_warning = False
        self._receive_lock = asyncio.Lock()  # Prevent concurrent receive operations

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            session = async_get_clientsession(self.hass, verify_ssl=self._verify_ssl)
            self._ws = await session.ws_connect(self._url, heartbeat=30)
            self._connected = True
            # Reset reconnection state on successful connect
            self._reconnect_start = None
            self._reconnect_delay = RECONNECT_INITIAL_DELAY
            self._reconnect_logged_warning = False
            _LOGGER.info("WebSocket connected to LangChain server at %s", self._url)
            return True
        except aiohttp.ClientError as e:
            _LOGGER.debug("WebSocket connection failed: %s", e)
            self._connected = False
            return False
        except Exception as e:
            _LOGGER.debug("Unexpected error connecting to WebSocket: %s", e)
            self._connected = False
            return False

    async def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnecting with backoff if needed.

        Returns True if connected, False if reconnection window expired.
        """
        if self.is_connected:
            return True

        now = time.monotonic()

        # Start reconnection window if not already started
        if self._reconnect_start is None:
            self._reconnect_start = now
            _LOGGER.warning("WebSocket disconnected, attempting to reconnect...")

        # Check if we've exceeded the reconnection window
        elapsed = now - self._reconnect_start
        if elapsed >= RECONNECT_MAX_DURATION:
            if not self._reconnect_logged_warning:
                _LOGGER.error(
                    "Reconnection window expired after %d seconds, giving up",
                    RECONNECT_MAX_DURATION
                )
                self._reconnect_logged_warning = True
            return False

        # Attempt to reconnect
        if await self.connect():
            _LOGGER.info("WebSocket reconnected after %.1f seconds", elapsed)
            return True

        # Log progress sparingly (only at certain intervals)
        if elapsed > 30 and not self._reconnect_logged_warning:
            _LOGGER.warning(
                "Still trying to reconnect (%.0fs elapsed, next attempt in %ds)",
                elapsed, self._reconnect_delay
            )

        # Wait with backoff before next attempt
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(
            self._reconnect_delay * RECONNECT_BACKOFF_FACTOR,
            RECONNECT_MAX_DELAY
        )

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

        Automatically attempts reconnection with backoff if disconnected.

        Args:
            conversation_id: Unique identifier for the conversation
            messages: Full chat history including tool calls and results
            tools: Available HA tools

        Returns:
            Response dict with type, content, and optionally tool_calls
        """
        # Use lock to prevent concurrent receive operations
        async with self._receive_lock:
            # Try to ensure we're connected (with reconnection if needed)
            if not await self.ensure_connected():
                raise ConnectionError("WebSocket not connected and reconnection failed")

            try:
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
                        self._connected = False
                        _LOGGER.debug("WebSocket closed while waiting for response")
                        raise ConnectionError("WebSocket closed unexpectedly")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        self._connected = False
                        _LOGGER.debug("WebSocket error: %s", self._ws.exception())
                        raise ConnectionError(f"WebSocket error: {self._ws.exception()}")

                raise ConnectionError("No response received")

            except (aiohttp.ClientError, ConnectionError):
                # Mark as disconnected so next call triggers reconnection
                self._connected = False
                raise

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
        async with self._receive_lock:
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
