import logging
import json
from typing import Dict, Any, Optional
from jsonschema import Draft7Validator, ValidationError
import jsonschema


def format_validation_error_for_agent(
    tool_name: str,
    args: Dict[str, Any],
    error: ValidationError,
    original_query: str
) -> str:
    """
    Convert jsonschema ValidationError into an agent-friendly error message.

    Args:
        tool_name: Name of the tool that failed validation
        args: The arguments that were provided
        error: The ValidationError from jsonschema
        original_query: The user's original query for context

    Returns:
        A formatted error message that helps the LLM correct its mistake
    """
    if error.path:
        field_path = ".".join(str(p) for p in error.path)
        field_name = list(error.path)[-1] if error.path else "unknown"
    else:
        field_path = "root"
        field_name = "root"

    invalid_value = error.instance

    if error.validator == "enum":
        allowed_values = error.validator_value
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error: Parameter '{field_name}' has invalid value: {json.dumps(invalid_value)}\n"
            f"Allowed values: {json.dumps(allowed_values)}\n\n"
            f"The value you provided ({json.dumps(invalid_value)}) is not in the list of allowed values.\n"
            f"Please call the tool again with a valid value from the allowed list, "
            f"or omit this parameter if it's not required."
        )
    elif error.validator == "type":
        expected_type = error.validator_value
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error: Parameter '{field_name}' has incorrect type.\n"
            f"Expected type: {expected_type}\n"
            f"Actual value: {json.dumps(invalid_value)}\n\n"
            f"Please call the tool again with the correct type."
        )
    elif error.validator == "required":
        missing_props = error.validator_value
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error: Required parameter(s) missing: {missing_props}\n\n"
            f"Please call the tool again with all required parameters."
        )
    else:
        error_msg = (
            f"Tool call validation failed for '{tool_name}'.\n\n"
            f"Original request: \"{original_query}\"\n\n"
            f"Error at '{field_path}': {error.message}\n"
            f"Provided value: {json.dumps(invalid_value)}\n\n"
            f"Please call the tool again with valid parameters."
        )

    return error_msg


def validate_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    tool_schema: Dict[str, Any],
    original_query: str,
    logger: logging.Logger
) -> Optional[str]:
    """
    Validate a tool call's arguments against its JSON Schema.

    Args:
        tool_name: Name of the tool being called
        args: Arguments provided to the tool
        tool_schema: The OpenAI function format tool schema
        original_query: User's original query for error context
        logger: Logger instance

    Returns:
        None if valid, or an agent-friendly error message if invalid
    """
    try:
        parameters_schema = tool_schema.get("function", {}).get("parameters", {})

        if not parameters_schema:
            logger.warning(f"[validate_tool_call] No parameters schema found for {tool_name}")
            return None

        jsonschema.validate(instance=args, schema=parameters_schema)

        logger.info(f"[validate_tool_call] ✓ {tool_name} arguments are valid")
        return None

    except ValidationError as e:
        logger.warning(f"[validate_tool_call] ✗ {tool_name} validation failed")
        logger.warning(f"[validate_tool_call]   Technical error: {e.message}")
        logger.warning(f"[validate_tool_call]   Path: {list(e.path)}")
        logger.warning(f"[validate_tool_call]   Validator: {e.validator}")
        logger.warning(f"[validate_tool_call]   Invalid value: {e.instance}")
        if e.validator == "enum":
            logger.warning(f"[validate_tool_call]   Allowed values: {e.validator_value}")

        friendly_error = format_validation_error_for_agent(
            tool_name=tool_name,
            args=args,
            error=e,
            original_query=original_query
        )

        logger.info(f"[validate_tool_call]   Formatted error for agent:\n{friendly_error}")

        return friendly_error
    except Exception as e:
        logger.error(f"[validate_tool_call] Unexpected error validating {tool_name}: {e}")
        return None
