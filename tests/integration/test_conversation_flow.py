"""Tests for multi-turn conversation context.

Hard assertions: state changes actually happened.
Soft assertions: response text quality via LLM judge.
"""
import logging

import pytest

from .helpers.ha_client import HAClient
from .helpers.llm_judge import LLMJudge
from .helpers.state_manager import StateManager

logger = logging.getLogger(__name__)


def _speech(result: dict) -> str:
    """Extract the plain-text speech from a conversation result."""
    return result["response"]["speech"]["plain"]["speech"]


class TestConversationFlow:
    """Verify multi-turn context is maintained via conversation_id."""

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_follow_up_command(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        state_manager: StateManager,
        llm_judge: LLMJudge,
    ):
        """Turn on a light, then send a follow-up using the same conversation."""
        await state_manager.verify_state("light.bed_light", "off")

        # First turn: turn on
        result1 = await ha_client.send_conversation(
            "turn on the bed light", agent_id=langchain_agent_id
        )
        conversation_id = result1.get("conversation_id")
        assert conversation_id, "First response should include a conversation_id"

        speech1 = _speech(result1)
        assert speech1, "First response should not be empty"

        # Hard: bed light must actually be on
        await ha_client.wait_for_state("light.bed_light", "on", timeout=30)

        # Second turn: follow-up dim (same conversation)
        result2 = await ha_client.send_conversation(
            "now dim it to 50 percent",
            conversation_id=conversation_id,
            agent_id=langchain_agent_id,
        )

        speech2 = _speech(result2)
        assert speech2, "Follow-up response should not be empty"

        # Hard: bed light should be on at ~50% brightness
        state = await ha_client.wait_for_state("light.bed_light", "on", timeout=30)
        brightness = state["attributes"].get("brightness", 0)
        assert 100 <= brightness <= 155, (
            f"Expected brightness ~128 (50%), got {brightness}"
        )

        # Soft: judge the follow-up response text
        states = await state_manager.get_all_light_states()
        judgment = await llm_judge.evaluate(
            user_command="now dim it to 50 percent (follow-up after turning on the bed light)",
            assistant_response=speech2,
            expected_behavior="The bed light should be dimmed to approximately 50% brightness",
            actual_states=states,
        )
        if judgment.passed:
            logger.info("Follow-up judge PASSED (score=%.1f)", judgment.score)
        else:
            logger.warning(
                "Follow-up judge FAILED (score=%.1f, criteria=%s): %s",
                judgment.score, judgment.criteria_scores, judgment.reasoning,
            )
