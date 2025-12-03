"""WebSocket handler for HA communication - thin routing layer."""
import logging
import uuid
from typing import Dict

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
conversation_logger = logging.getLogger('conversations')


class ConnectionManager:
    """Track active WebSocket connections."""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.connections[client_id] = websocket
        logger.info(f"Client connected: {client_id[:8]}")
        return client_id

    def disconnect(self, client_id: str):
        self.connections.pop(client_id, None)
        logger.info(f"Client disconnected: {client_id[:8]}")


class WebSocketHandler:
    """Handle WebSocket messages from HA - thin routing layer.

    This handler does NOT parse or manipulate messages.
    It simply routes them to the correct agent by conversation_id.
    """

    def __init__(self, router_agent):
        self.router_agent = router_agent
        self.manager = ConnectionManager()

    async def handle_connection(self, websocket: WebSocket):
        client_id = await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                await self._route_message(websocket, client_id, data)
        except WebSocketDisconnect:
            pass
        finally:
            self.manager.disconnect(client_id)

    async def _route_message(self, ws: WebSocket, client_id: str, data: dict):
        msg_type = data.get("type")

        if msg_type == "ping":
            await ws.send_json({"type": "pong"})

        elif msg_type == "conversation":
            await self._handle_conversation(ws, data)

        elif msg_type == "log":
            self._handle_log(data)

    async def _handle_conversation(self, ws: WebSocket, data: dict):
        """Pass conversation straight through to router_agent."""
        conv_id = data.get("conversation_id", "")
        messages = data.get("messages", [])
        tools = data.get("tools", [])

        conversation_logger.info(f"{'='*60}")
        conversation_logger.info(f"CONVERSATION | {conv_id[:8]}... | {len(messages)} messages")

        try:
            # Pass messages directly to router_agent - no extraction/manipulation
            result = await self.router_agent.process(
                messages=messages,
                tools=tools,
                conversation_id=conv_id
            )

            # Forward response back to HA
            await ws.send_json({
                "type": result.get("type", "response"),
                "conversation_id": conv_id,
                **{k: v for k, v in result.items() if k not in ("type", "conversation_id")}
            })

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            await ws.send_json({
                "type": "error",
                "conversation_id": conv_id,
                "code": "processing_error",
                "message": str(e)
            })

    def _handle_log(self, data: dict):
        level = data.get("level", "INFO").upper()
        message = data.get("message", "")
        log_level = getattr(logging, level, logging.INFO)
        logger.info(f"Received HA log: {level} - {message[:50]}...")
        logging.getLogger("home_assistant").log(log_level, f"[HA] {message}")
