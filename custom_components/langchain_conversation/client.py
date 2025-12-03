"""WebSocket client for LangChain server."""
import asyncio
import json
import logging
import queue
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
    """Logging handler that forwards logs to LangChain server via WebSocket.

    Uses a thread-safe queue since emit() is called synchronously from logging
    which may happen from different threads.
    """

    def __init__(self, client: "LangChainClient"):
        super().__init__()
        self.client = client
        self._queue: queue.Queue[Dict[str, str]] = queue.Queue(maxsize=100)
        self._task: Optional[asyncio.Task] = None

    def emit(self, record: logging.LogRecord):
        """Queue log record for async sending (thread-safe)."""
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
            except queue.Full:
                pass  # Drop if queue is full
            except Exception:
                pass  # Best effort - don't break logging

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
                # Use non-blocking get with small sleep to allow cancellation
                try:
                    log_entry = self._queue.get_nowait()
                    if self.client.is_connected:
                        await self.client.send_log(
                            log_entry["level"],
                            f"[{log_entry['logger']}] {log_entry['message']}"
                        )
                except queue.Empty:
                    await asyncio.sleep(0.1)
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

        # Dedicated reader task pattern
        self._reader_task: Optional[asyncio.Task] = None
        self._response_queues: Dict[str, asyncio.Queue[Dict[str, Any]]] = {}
        self._queues_lock = asyncio.Lock()
        self._pending_ping: Optional[asyncio.Future[bool]] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed

    async def connect(self) -> bool:
        """Establish WebSocket connection and start reader task."""
        try:
            session = async_get_clientsession(self.hass, verify_ssl=self._verify_ssl)
            self._ws = await session.ws_connect(self._url, heartbeat=30)
            self._connected = True
            await self._start_reader_task()
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
        await self._stop_reader_task()
        async with self._queues_lock:
            self._response_queues.clear()
        if self._ws:
            await self._ws.close()
            self._ws = None
            _LOGGER.info("WebSocket disconnected from LangChain server")

    async def _start_reader_task(self) -> None:
        """Start the background reader task."""
        if self._reader_task is None or self._reader_task.done():
            self._reader_task = asyncio.create_task(self._reader_loop())

    async def _stop_reader_task(self) -> None:
        """Stop the background reader task."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

    async def _reader_loop(self) -> None:
        """Background task that reads all WebSocket messages and routes them."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._route_message(data)
                    except json.JSONDecodeError:
                        pass
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            self._connected = False
            await self._notify_all_waiters_of_disconnect()

    async def _route_message(self, data: Dict[str, Any]) -> None:
        """Route incoming message to appropriate handler."""
        msg_type = data.get("type")

        if msg_type == "pong":
            if self._pending_ping and not self._pending_ping.done():
                self._pending_ping.set_result(True)
        elif msg_type in ("response", "tool_call", "error"):
            conv_id = data.get("conversation_id")
            if conv_id:
                async with self._queues_lock:
                    queue = self._response_queues.get(conv_id)
                    if queue:
                        await queue.put(data)

    async def _notify_all_waiters_of_disconnect(self) -> None:
        """Notify all waiting operations that connection was lost."""
        if self._pending_ping and not self._pending_ping.done():
            self._pending_ping.set_exception(ConnectionError("WebSocket disconnected"))

        async with self._queues_lock:
            disconnect_msg: Dict[str, Any] = {
                "type": "error",
                "code": "connection_lost",
                "message": "WebSocket connection lost"
            }
            for queue in self._response_queues.values():
                try:
                    await queue.put(disconnect_msg)
                except Exception:
                    pass

    async def send_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send conversation and wait for response via queue.

        This method is concurrency-safe - multiple conversations can be
        in flight simultaneously, each waiting on their own queue.

        Args:
            conversation_id: Unique identifier for the conversation
            messages: Full chat history including tool calls and results
            tools: Available HA tools

        Returns:
            Response dict with type, content, and optionally tool_calls
        """
        if not await self.ensure_connected():
            raise ConnectionError("WebSocket not connected and reconnection failed")

        response_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1)

        async with self._queues_lock:
            self._response_queues[conversation_id] = response_queue

        try:
            await self._ws.send_json({
                "type": "conversation",
                "conversation_id": conversation_id,
                "messages": messages,
                "tools": tools
            })

            response = await response_queue.get()

            if response.get("code") == "connection_lost":
                raise ConnectionError("WebSocket connection lost while waiting for response")

            return response

        except (aiohttp.ClientError, ConnectionError):
            self._connected = False
            raise
        finally:
            async with self._queues_lock:
                self._response_queues.pop(conversation_id, None)

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
        """Send ping and wait for pong via Future."""
        if not self.is_connected:
            return False

        loop = asyncio.get_running_loop()
        self._pending_ping = loop.create_future()

        try:
            await self._ws.send_json({"type": "ping"})
            async with asyncio.timeout(10):
                return await self._pending_ping
        except (TimeoutError, asyncio.TimeoutError):
            return False
        except Exception:
            return False
        finally:
            self._pending_ping = None
