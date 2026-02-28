"""LLM-as-judge evaluator for response quality."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating a voice assistant's response to a smart-home command.

## User command
{user_command}

## Expected behavior
{expected_behavior}

## Actual entity states after command
{actual_states}

## Assistant response
{assistant_response}

---

Score EACH criterion from 1 (terrible) to 5 (excellent):

1. **Accuracy** — Does the response match what actually happened to the devices?
2. **Naturalness** — Is it suitable for a voice assistant (conversational, not robotic)?
3. **Completeness** — Does it address the full request?
4. **Conciseness** — Is it appropriately brief for voice output?

Return ONLY valid JSON (no markdown fences) with this schema:
{{"accuracy": <int>, "naturalness": <int>, "completeness": <int>, "conciseness": <int>, "reasoning": "<brief explanation>"}}
"""


@dataclass
class JudgmentResult:
    passed: bool
    score: float
    reasoning: str
    criteria_scores: dict[str, int]


class LLMJudge:
    """Evaluate assistant responses using a local LLM."""

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0)

    async def evaluate(
        self,
        user_command: str,
        assistant_response: str,
        expected_behavior: str,
        actual_states: dict | str,
    ) -> JudgmentResult:
        """Run the judge and return a structured result."""
        if isinstance(actual_states, dict):
            actual_states = json.dumps(actual_states, indent=2)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            user_command=user_command,
            expected_behavior=expected_behavior,
            assistant_response=assistant_response,
            actual_states=actual_states,
        )

        try:
            response = await self.llm.ainvoke(prompt)
            return self._parse(response.content)
        except Exception as exc:
            logger.error("LLM judge failed: %s", exc)
            return JudgmentResult(
                passed=False,
                score=0.0,
                reasoning=f"Judge error: {exc}",
                criteria_scores={},
            )

    @staticmethod
    def _parse(raw: str) -> JudgmentResult:
        """Extract the JSON scores from the LLM output."""
        # Try to find JSON object in the response
        json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if not json_match:
            return JudgmentResult(
                passed=False,
                score=0.0,
                reasoning=f"Could not parse judge output: {raw[:200]}",
                criteria_scores={},
            )

        data = json.loads(json_match.group())
        criteria = {
            k: int(data[k])
            for k in ("accuracy", "naturalness", "completeness", "conciseness")
            if k in data
        }
        avg = sum(criteria.values()) / len(criteria) if criteria else 0
        passed = criteria.get("accuracy", 0) >= 3 and avg >= 3.0

        return JudgmentResult(
            passed=passed,
            score=round(avg, 2),
            reasoning=data.get("reasoning", ""),
            criteria_scores=criteria,
        )
