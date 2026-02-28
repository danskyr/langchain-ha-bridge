"""Tests for error/edge-case handling: unknown devices, ambiguous commands."""
import pytest

from .helpers.ha_client import HAClient
from .helpers.llm_judge import LLMJudge


def _speech(result: dict) -> str:
    """Extract the plain-text speech from a conversation result."""
    return result["response"]["speech"]["plain"]["speech"]


class TestErrorHandling:
    """Verify the system handles edge cases gracefully."""

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_unknown_device(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        llm_judge: LLMJudge,
    ):
        """Asking about a device that doesn't exist should not crash."""
        result = await ha_client.send_conversation(
            "turn on the garage door light", agent_id=langchain_agent_id
        )

        speech = _speech(result)
        assert speech, "Should still get a response for unknown devices"

        judgment = await llm_judge.evaluate(
            user_command="turn on the garage door light",
            assistant_response=speech,
            expected_behavior="The system should indicate it cannot find a garage door light, explain the device is not available, or attempt the action and report the outcome",
            actual_states="No garage door light entity exists",
        )
        assert judgment.passed, (
            f"LLM judge failed: {judgment.reasoning} "
            f"(score={judgment.score}, criteria={judgment.criteria_scores})"
        )

    @pytest.mark.flaky(reruns=1, reruns_delay=2)
    async def test_ambiguous_command(
        self,
        ha_client: HAClient,
        langchain_agent_id: str,
        llm_judge: LLMJudge,
    ):
        """An ambiguous command should get a reasonable response (not crash)."""
        result = await ha_client.send_conversation(
            "turn on the light", agent_id=langchain_agent_id
        )

        speech = _speech(result)
        assert speech, "Should still get a response for ambiguous commands"

        judgment = await llm_judge.evaluate(
            user_command="turn on the light",
            assistant_response=speech,
            expected_behavior="The system should respond reasonably: ask which light, turn on a default light, list available lights, or attempt the action. Any coherent response that doesn't crash is acceptable.",
            actual_states="Multiple lights exist in the demo home",
        )
        assert judgment.passed, (
            f"LLM judge failed: {judgment.reasoning} "
            f"(score={judgment.score}, criteria={judgment.criteria_scores})"
        )
