"""Tests for langchain_agent.src.utils.text module."""
import pytest
from langchain_agent.src.utils.text import preview_text


class TestPreviewText:
    """Tests for the preview_text function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert preview_text("") == ""

    def test_none_like_empty(self):
        """Empty-like values should return empty string."""
        assert preview_text("") == ""

    def test_short_text_unchanged(self):
        """Text shorter than max_len should be returned unchanged."""
        text = "Hello, world!"
        assert preview_text(text) == text

    def test_exact_length_unchanged(self):
        """Text exactly max_len should be returned unchanged."""
        text = "a" * 600
        assert preview_text(text, max_len=600) == text

    def test_long_text_truncated(self):
        """Text longer than max_len should be truncated with ellipsis."""
        text = "a" * 1000
        result = preview_text(text, max_len=100)
        assert " ... " in result
        assert len(result) < len(text)

    def test_truncation_shows_start_and_end(self):
        """Truncated text should show both start and end."""
        text = "START" + "x" * 1000 + "END"
        result = preview_text(text, max_len=100)
        assert result.startswith("START")
        assert result.endswith("END")

    def test_custom_max_len(self):
        """Custom max_len should be respected."""
        text = "a" * 100
        result = preview_text(text, max_len=50)
        assert " ... " in result
        assert len(result) <= 50

    def test_small_max_len(self):
        """Very small max_len should still work."""
        text = "Hello, world! This is a test."
        result = preview_text(text, max_len=20)
        assert " ... " in result

    def test_preserves_special_characters(self):
        """Special characters should be preserved."""
        text = "Hello! こんにちは! 🎉" + "x" * 600
        result = preview_text(text)
        assert "Hello! こんにちは! 🎉" in result

    def test_newlines_preserved(self):
        """Newlines should be preserved in output."""
        text = "Line1\nLine2\n" + "x" * 600 + "\nLastLine"
        result = preview_text(text)
        assert "\n" in result
