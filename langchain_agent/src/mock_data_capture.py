import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class MockDataCapture:
    def __init__(self, base_dir: str = "logs/mock_data"):
        self.enabled = os.getenv("CAPTURE_MOCK_DATA", "false").lower() == "true"
        self.base_dir = Path(base_dir)
        self._sequence_counters: Dict[str, int] = {}

    def _get_conversation_dir(self, conversation_id: str) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        conv_dir = self.base_dir / today / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        return conv_dir

    def _get_next_sequence(self, conversation_id: str) -> int:
        if conversation_id not in self._sequence_counters:
            conv_dir = self._get_conversation_dir(conversation_id)
            existing = list(conv_dir.glob("*_request.json"))
            self._sequence_counters[conversation_id] = len(existing) + 1
        seq = self._sequence_counters[conversation_id]
        self._sequence_counters[conversation_id] += 1
        return seq

    def capture_request(self, conversation_id: str, request_data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        conv_dir = self._get_conversation_dir(conversation_id)
        seq = self._get_next_sequence(conversation_id)
        filepath = conv_dir / f"{seq:03d}_request.json"
        with open(filepath, "w") as f:
            json.dump(request_data, f, indent=2, default=str)

    def capture_response(self, conversation_id: str, response_data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        conv_dir = self._get_conversation_dir(conversation_id)
        seq = self._sequence_counters.get(conversation_id, 1) - 1
        filepath = conv_dir / f"{seq:03d}_response.json"
        with open(filepath, "w") as f:
            json.dump(response_data, f, indent=2, default=str)


mock_capture = MockDataCapture()
