"""Tests for light on/off, brightness, and color commands through the full pipeline.

Each test:
  1. Sends a natural language command through HA's conversation API
  2. Hard-asserts the expected device state change actually happened
  3. Soft-evaluates the response text quality via LLM judge (logged, not asserted)
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


async def _judge_and_log(
    llm_judge: LLMJudge,
    user_command: str,
    assistant_response: str,
    expected_behavior: str,
    actual_states,
):
    """Run the LLM judge and log the result (does not assert)."""
    judgment = await llm_judge.evaluate(
        user_command=user_command,
        assistant_response=assistant_response,
        expected_behavior=expected_behavior,
        actual_states=actual_states,
    )
    if judgment.passed:
        logger.info("LLM judge PASSED (score=%.1f): %s", judgment.score, judgment.reasoning)
    else:
        logger.warning(
            "LLM judge FAILED (score=%.1f, criteria=%s): %s",
            judgment.score, judgment.criteria_scores, judgment.reasoning,
        )


# ---------------------------------------------------------------------------
# On / Off
# ---------------------------------------------------------------------------
class TestLightOnOff:
    """Verify basic on/off commands flow through the full pipeline."""

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_turn_off_kitchen_lights(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        state_manager: StateManager,
        llm_judge: LLMJudge,
    ):
        await state_manager.verify_state("light.kitchen_lights", "on")

        result = await ha_client.send_conversation(
            "turn off the kitchen lights", agent_id=langchain_agent_id
        )

        speech = _speech(result)
        assert speech, "Response speech should not be empty"

        # Hard: the kitchen lights must actually be off
        await ha_client.wait_for_state("light.kitchen_lights", "off", timeout=30)

        # Soft: judge the response text
        states = await state_manager.get_all_light_states()
        await _judge_and_log(
            llm_judge, "turn off the kitchen lights", speech,
            "The kitchen lights should be turned off", states,
        )

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_turn_on_bed_light(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        state_manager: StateManager,
        llm_judge: LLMJudge,
    ):
        await state_manager.verify_state("light.bed_light", "off")

        result = await ha_client.send_conversation(
            "turn on the bed light", agent_id=langchain_agent_id
        )

        speech = _speech(result)
        assert speech, "Response speech should not be empty"

        # Hard: the bed light must actually be on
        await ha_client.wait_for_state("light.bed_light", "on", timeout=30)

        # Soft: judge the response text
        states = await state_manager.get_all_light_states()
        await _judge_and_log(
            llm_judge, "turn on the bed light", speech,
            "The bed light should be turned on", states,
        )

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_turn_off_all_lights(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        state_manager: StateManager,
        llm_judge: LLMJudge,
    ):
        result = await ha_client.send_conversation(
            "turn off all the lights", agent_id=langchain_agent_id
        )

        speech = _speech(result)
        assert speech, "Response speech should not be empty"

        # Hard: every demo light must actually be off
        for entity_id in (
            "light.bed_light",
            "light.ceiling_lights",
            "light.kitchen_lights",
            "light.office_rgbw_lights",
            "light.living_room_rgbww_lights",
            "light.entrance_color_white_lights",
        ):
            await ha_client.wait_for_state(entity_id, "off", timeout=30)

        # Soft: judge the response text
        states = await state_manager.get_all_light_states()
        await _judge_and_log(
            llm_judge, "turn off all the lights", speech,
            "All lights in the home should be turned off", states,
        )


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------
class TestLightBrightness:
    """Verify brightness adjustment commands."""

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_dim_ceiling_lights(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        state_manager: StateManager,
        llm_judge: LLMJudge,
    ):
        await state_manager.verify_state("light.ceiling_lights", "on")

        result = await ha_client.send_conversation(
            "dim the ceiling lights to 50 percent", agent_id=langchain_agent_id
        )

        speech = _speech(result)
        assert speech, "Response speech should not be empty"

        # Hard: brightness should be approximately 128 (50% of 255)
        state = await ha_client.wait_for_state("light.ceiling_lights", "on", timeout=30)
        brightness = state["attributes"].get("brightness", 0)
        assert 100 <= brightness <= 155, (
            f"Expected brightness ~128 (50%), got {brightness}"
        )

        # Soft: judge the response text
        states = await state_manager.get_all_light_states()
        await _judge_and_log(
            llm_judge, "dim the ceiling lights to 50 percent", speech,
            "The ceiling lights should be dimmed to approximately 50% brightness", states,
        )


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------
class TestLightColor:
    """Verify color change commands on RGBW lights."""

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_set_office_lights_red(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        state_manager: StateManager,
        llm_judge: LLMJudge,
    ):
        await state_manager.verify_state("light.office_rgbw_lights", "on")

        result = await ha_client.send_conversation(
            "set the office lights to red", agent_id=langchain_agent_id
        )

        speech = _speech(result)
        assert speech, "Response speech should not be empty"

        # Hard: light should still be on with a reddish color
        state = await ha_client.wait_for_state("light.office_rgbw_lights", "on", timeout=30)
        rgb = state["attributes"].get("rgb_color")
        if rgb:
            assert rgb[0] > rgb[1] and rgb[0] > rgb[2], (
                f"Expected red-dominant color, got RGB {rgb}"
            )

        # Soft: judge the response text
        states = await state_manager.get_all_light_states()
        await _judge_and_log(
            llm_judge, "set the office lights to red", speech,
            "The office lights should be set to a red color", states,
        )
