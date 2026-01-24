"""Tests for langchain_agent.src.mock_data_capture module."""
import os
import json
import pytest
from datetime import datetime
from unittest.mock import patch
from langchain_agent.src.mock_data_capture import MockDataCapture


class TestMockDataCapture:
    """Tests for the MockDataCapture class."""

    def test_init_disabled_by_default(self):
        """MockDataCapture should be disabled by default."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "false"}):
            capture = MockDataCapture()
            assert capture.enabled is False

    def test_init_enabled_by_env(self):
        """MockDataCapture should be enabled when env var is true."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "true"}):
            capture = MockDataCapture()
            assert capture.enabled is True

    def test_custom_base_dir(self):
        """Custom base directory should be respected."""
        capture = MockDataCapture(base_dir="/custom/path")
        assert str(capture.base_dir) == "/custom/path"

    def test_capture_request_disabled(self, temp_dir):
        """Capture should do nothing when disabled."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "false"}):
            capture = MockDataCapture(base_dir=str(temp_dir))
            capture.capture_request("conv-123", {"test": "data"})
            # No files should be created
            assert len(list(temp_dir.rglob("*.json"))) == 0

    def test_capture_request_enabled(self, temp_dir):
        """Capture should write request file when enabled."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "true"}):
            capture = MockDataCapture(base_dir=str(temp_dir))
            capture.capture_request("conv-123", {"prompt": "test query"})

            # Find the created file
            json_files = list(temp_dir.rglob("*_request.json"))
            assert len(json_files) == 1

            # Verify content
            with open(json_files[0]) as f:
                data = json.load(f)
            assert data["prompt"] == "test query"

    def test_capture_response_enabled(self, temp_dir):
        """Capture should write response file when enabled."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "true"}):
            capture = MockDataCapture(base_dir=str(temp_dir))
            capture.capture_request("conv-123", {"prompt": "test"})
            capture.capture_response("conv-123", {"response": "test response"})

            # Find the created files
            request_files = list(temp_dir.rglob("*_request.json"))
            response_files = list(temp_dir.rglob("*_response.json"))
            assert len(request_files) == 1
            assert len(response_files) == 1

    def test_sequence_numbering(self, temp_dir):
        """Multiple captures should have sequential numbers."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "true"}):
            capture = MockDataCapture(base_dir=str(temp_dir))
            capture.capture_request("conv-123", {"req": 1})
            capture.capture_response("conv-123", {"resp": 1})
            capture.capture_request("conv-123", {"req": 2})
            capture.capture_response("conv-123", {"resp": 2})

            request_files = sorted(temp_dir.rglob("*_request.json"))
            assert len(request_files) == 2
            assert "001_request" in str(request_files[0])
            assert "002_request" in str(request_files[1])

    def test_conversation_dir_structure(self, temp_dir):
        """Files should be organized by date and conversation."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "true"}):
            capture = MockDataCapture(base_dir=str(temp_dir))
            capture.capture_request("my-conversation", {"test": "data"})

            # Check directory structure
            today = datetime.now().strftime("%Y-%m-%d")
            expected_dir = temp_dir / today / "my-conversation"
            assert expected_dir.exists()

    def test_multiple_conversations(self, temp_dir):
        """Different conversations should have separate directories."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "true"}):
            capture = MockDataCapture(base_dir=str(temp_dir))
            capture.capture_request("conv-1", {"data": 1})
            capture.capture_request("conv-2", {"data": 2})

            today = datetime.now().strftime("%Y-%m-%d")
            assert (temp_dir / today / "conv-1").exists()
            assert (temp_dir / today / "conv-2").exists()

    def test_json_serialization_with_special_types(self, temp_dir):
        """Capture should handle non-JSON-serializable types via default=str."""
        with patch.dict(os.environ, {"CAPTURE_MOCK_DATA": "true"}):
            capture = MockDataCapture(base_dir=str(temp_dir))

            # datetime is not JSON-serializable by default
            data = {
                "timestamp": datetime.now(),
                "bytes": b"test bytes"
            }
            capture.capture_request("conv-123", data)

            # Should not raise, file should be created
            json_files = list(temp_dir.rglob("*.json"))
            assert len(json_files) == 1
