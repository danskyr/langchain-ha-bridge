"""Async Home Assistant REST API wrapper for integration tests."""
import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)


class HAClient:
    """Async wrapper around the Home Assistant REST API."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=aiohttp.ClientTimeout(total=90),
            )
        return self._session

    async def get_state(self, entity_id: str) -> dict:
        """Get the state of a single entity."""
        async with self.session.get(f"{self.base_url}/api/states/{entity_id}") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_all_states(self) -> list[dict]:
        """Get all entity states."""
        async with self.session.get(f"{self.base_url}/api/states") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def call_service(self, domain: str, service: str, data: dict | None = None) -> list[dict]:
        """Call a Home Assistant service."""
        async with self.session.post(
            f"{self.base_url}/api/services/{domain}/{service}",
            json=data or {},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def send_conversation(
        self,
        text: str,
        conversation_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict:
        """Send a text command through the HA conversation API."""
        payload: dict = {"text": text, "language": "en"}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if agent_id:
            payload["agent_id"] = agent_id

        logger.info("Sending conversation: %s", text)
        async with self.session.post(
            f"{self.base_url}/api/conversation/process",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()
            logger.info("Conversation response: %s", result)
            return result

    async def wait_for_state(
        self,
        entity_id: str,
        expected_state: str,
        timeout: float = 30,
        poll_interval: float = 1,
    ) -> dict:
        """Poll until an entity reaches the expected state or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        last_state = None
        while asyncio.get_event_loop().time() < deadline:
            state = await self.get_state(entity_id)
            last_state = state
            if state["state"] == expected_state:
                return state
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"{entity_id} did not reach state '{expected_state}' within {timeout}s. "
            f"Last state: {last_state['state'] if last_state else 'unknown'}"
        )

    async def close(self):
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
