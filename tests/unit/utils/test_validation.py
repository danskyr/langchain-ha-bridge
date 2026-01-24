"""Tests for langchain_agent.src.utils.validation module."""
import pytest
import logging
from jsonschema import ValidationError
from langchain_agent.src.utils.validation import (
    format_validation_error_for_agent,
    validate_tool_call
)


class TestFormatValidationErrorForAgent:
    """Tests for the format_validation_error_for_agent function."""

    def test_enum_error_formatting(self):
        """Enum validation errors should be formatted with allowed values."""
        error = ValidationError(
            message="'invalid_value' is not one of ['light', 'switch']",
            validator="enum",
            path=["domain"],
            validator_value=["light", "switch"],
            instance="invalid_value"
        )

        result = format_validation_error_for_agent(
            tool_name="HassTurnOn",
            args={"domain": "invalid_value"},
            error=error,
            original_query="turn on the light"
        )

        assert "HassTurnOn" in result
        assert "domain" in result
        assert "invalid_value" in result
        assert "light" in result
        assert "switch" in result
        assert "Allowed values" in result

    def test_type_error_formatting(self):
        """Type validation errors should show expected type."""
        error = ValidationError(
            message="123 is not of type 'string'",
            validator="type",
            path=["area"],
            validator_value="string",
            instance=123
        )

        result = format_validation_error_for_agent(
            tool_name="HassTurnOn",
            args={"area": 123},
            error=error,
            original_query="turn on the light"
        )

        assert "HassTurnOn" in result
        assert "area" in result
        assert "string" in result
        assert "incorrect type" in result

    def test_required_error_formatting(self):
        """Required field errors should list missing fields."""
        error = ValidationError(
            message="'name' is a required property",
            validator="required",
            path=[],
            validator_value=["name"],
            instance={}
        )

        result = format_validation_error_for_agent(
            tool_name="HassTurnOn",
            args={},
            error=error,
            original_query="turn on the light"
        )

        assert "HassTurnOn" in result
        assert "Required parameter" in result or "required" in result.lower()

    def test_generic_error_formatting(self):
        """Other validation errors should still be formatted."""
        error = ValidationError(
            message="Some other error",
            validator="custom",
            path=["field"],
            validator_value=None,
            instance="value"
        )

        result = format_validation_error_for_agent(
            tool_name="HassTurnOn",
            args={"field": "value"},
            error=error,
            original_query="turn on the light"
        )

        assert "HassTurnOn" in result
        assert "Some other error" in result

    def test_includes_original_query(self):
        """Error message should include the original query."""
        error = ValidationError(
            message="test error",
            validator="type",
            path=["test"],
            validator_value="string",
            instance=123
        )

        result = format_validation_error_for_agent(
            tool_name="TestTool",
            args={},
            error=error,
            original_query="my specific query"
        )

        assert "my specific query" in result

    def test_root_path_handling(self):
        """Errors at root path should be handled gracefully."""
        error = ValidationError(
            message="Root level error",
            validator="type",
            path=[],
            validator_value="object",
            instance="not_an_object"
        )

        result = format_validation_error_for_agent(
            tool_name="TestTool",
            args={},
            error=error,
            original_query="test"
        )

        assert "root" in result


class TestValidateToolCall:
    """Tests for the validate_tool_call function."""

    def test_valid_call_returns_none(self, mock_logger, sample_tool_schema):
        """Valid tool calls should return None."""
        result = validate_tool_call(
            tool_name="HassTurnOn",
            args={"area": "living room"},
            tool_schema=sample_tool_schema,
            original_query="turn on the light",
            logger=mock_logger
        )
        assert result is None

    def test_invalid_enum_returns_error(self, mock_logger, sample_tool_schema):
        """Invalid enum values should return error message."""
        result = validate_tool_call(
            tool_name="HassTurnOn",
            args={"device_class": "invalid_class"},
            tool_schema=sample_tool_schema,
            original_query="turn on the light",
            logger=mock_logger
        )
        assert result is not None
        assert "invalid_class" in result

    def test_invalid_type_returns_error(self, mock_logger, sample_tool_schema):
        """Invalid types should return error message."""
        result = validate_tool_call(
            tool_name="HassTurnOn",
            args={"area": 123},  # should be string
            tool_schema=sample_tool_schema,
            original_query="turn on the light",
            logger=mock_logger
        )
        assert result is not None
        assert "type" in result.lower()

    def test_empty_schema_returns_none(self, mock_logger):
        """Missing schema should return None (no validation)."""
        schema = {"function": {}}
        result = validate_tool_call(
            tool_name="TestTool",
            args={"anything": "goes"},
            tool_schema=schema,
            original_query="test",
            logger=mock_logger
        )
        assert result is None

    def test_valid_array_parameter(self, mock_logger, sample_tool_schema):
        """Valid array parameters should pass validation."""
        result = validate_tool_call(
            tool_name="HassTurnOn",
            args={"domain": ["light", "switch"]},
            tool_schema=sample_tool_schema,
            original_query="turn on lights and switches",
            logger=mock_logger
        )
        assert result is None

    def test_invalid_array_item(self, mock_logger, sample_tool_schema):
        """Invalid array item enum should return error."""
        result = validate_tool_call(
            tool_name="HassTurnOn",
            args={"domain": ["light", "invalid_domain"]},
            tool_schema=sample_tool_schema,
            original_query="turn on the light",
            logger=mock_logger
        )
        assert result is not None
        assert "invalid_domain" in result

    def test_empty_args_valid_if_no_required(self, mock_logger, sample_tool_schema):
        """Empty args should be valid if no required fields."""
        result = validate_tool_call(
            tool_name="HassTurnOn",
            args={},
            tool_schema=sample_tool_schema,
            original_query="turn on",
            logger=mock_logger
        )
        assert result is None
