"""Manages demo entity states for test isolation."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ha_client import HAClient

logger = logging.getLogger(__name__)

# Demo integration initial states for lights.
# The demo: integration seeds these entities automatically.
DEMO_LIGHT_INITIAL_STATES: dict[str, str] = {
    "light.bed_light": "off",
    "light.ceiling_lights": "on",
    "light.kitchen_lights": "on",
    "light.office_rgbw_lights": "on",
    "light.living_room_rgbww_lights": "on",
    "light.entrance_color_white_lights": "on",
}


class StateManager:
    """Reset and verify demo entity states between tests."""

    def __init__(self, client: HAClient):
        self.client = client

    async def reset_all_lights(self):
        """Restore every demo light to its initial on/off state."""
        for entity_id, initial_state in DEMO_LIGHT_INITIAL_STATES.items():
            service = "turn_on" if initial_state == "on" else "turn_off"
            try:
                await self.client.call_service(
                    "light", service, {"entity_id": entity_id}
                )
            except Exception:
                logger.warning("Failed to reset %s – it may not exist yet", entity_id)

    async def verify_state(self, entity_id: str, expected_state: str, **attrs):
        """Assert that an entity matches the expected state and optional attributes."""
        state = await self.client.get_state(entity_id)
        assert state["state"] == expected_state, (
            f"{entity_id}: expected state '{expected_state}', got '{state['state']}'"
        )
        for key, value in attrs.items():
            actual = state["attributes"].get(key)
            assert actual == value, (
                f"{entity_id} attribute '{key}': expected {value}, got {actual}"
            )

    async def get_all_light_states(self) -> dict[str, dict]:
        """Return a dict of entity_id → {state, attributes} for demo lights."""
        result = {}
        for entity_id in DEMO_LIGHT_INITIAL_STATES:
            try:
                state = await self.client.get_state(entity_id)
                result[entity_id] = {
                    "state": state["state"],
                    "attributes": state.get("attributes", {}),
                }
            except Exception:
                result[entity_id] = {"state": "unknown", "attributes": {}}
        return result
