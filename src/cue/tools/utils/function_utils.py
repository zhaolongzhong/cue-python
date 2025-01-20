from typing import Any


def get_definition_by_model(func: dict[str, Any], model: str) -> dict[str, Any]:
    """
    Convert function definition JSON based on model requirements.

    - For Claude models, converts to input_schema format.
    - For other models, maintains original format or converts from input_schema if needed.

    Args:
        func (Dict[str, Any]): The function definition dictionary.
        model (str): The model name/identifier.

    Returns:
        Dict[str, Any]: Converted function definition matching model requirements.
    """
    is_claude = "claude" in model.lower()

    if is_claude:
        # If already in Claude format, return as is
        if "input_schema" in func:
            return func

        # Extract nested function data if present
        function_data = func.get("function")
        if function_data:
            return {
                "name": function_data.get("name"),
                "description": function_data.get("description"),
                "input_schema": function_data.get("parameters"),
            }

        # Handle flat structure
        return {
            "name": func.get("name"),
            "description": func.get("description"),
            "input_schema": func.get("parameters"),
        }
    else:
        # For non-Claude models
        if "input_schema" not in func:
            return func

        # Convert from Claude format to standard format
        return {
            "type": "function",  # Required for OpenAI models
            "function": {
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("input_schema"),
            },
        }
