# Reference: https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/tools/base.py
import sys
import json
from abc import ABCMeta, abstractmethod
from typing import Any, Optional
from pathlib import Path
from dataclasses import fields, replace, dataclass

from .utils.function_utils import get_definition_by_model
from .utils.function_to_json import function_to_json


@dataclass(frozen=True)
class ToolResult:
    output: Optional[str] = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None
    agent_transfer: Optional[Any] = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(field: str | None, other_field: str | None, concatenate: bool = True):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class BaseTool(metaclass=ABCMeta):
    """Abstract base class for tools."""

    def __init__(self):
        self._function = None  # Main function for the tool

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Executes the tool with the given arguments."""
        ...

    def to_json(self, model: str = "", use_file_definition: bool = True) -> dict:
        """Load or generate JSON parameters for the tool.

        Args:
            model: Optional model name for model-specific parameters
            use_file_definition: If True, prioritize loading from JSON file. If False, prefer dynamic generation.

        Returns:
            Dict containing the tool's parameters

        """
        # Get the current tool's directory and JSON file path
        module_path = sys.modules[self.__class__.__module__].__file__
        current_dir = Path(module_path).parent
        json_file_path = current_dir / f"{self.name}.json"
        json_file_path = json_file_path.resolve()

        # Function to generate parameters dynamically
        def generate_params():
            if not self._function:
                self._function = getattr(self, self.name, None)
                if not self._function:
                    raise ValueError(f"No function found with name {self.name} in {self.__class__.__name__}")

            params = function_to_json(self._function)

            # Save the generated JSON in cache directory
            cache_dir = current_dir / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file_path = cache_dir / f"{self.name}.json"
            with open(cache_file_path, "w") as f:
                json.dump(params, f, indent=2)

            return params

        if use_file_definition:
            # When use_file_definition is True, try to load from file first
            try:
                with open(json_file_path) as f:
                    return get_definition_by_model(json.load(f), model)
            except FileNotFoundError:
                # Fall back to dynamic generation if file doesn't exist
                params = generate_params()
                return get_definition_by_model(params, model)
        else:
            # When use_file_definition is False, prefer dynamic generation
            try:
                params = generate_params()
                return get_definition_by_model(params, model)
            except Exception as e:
                # Fall back to JSON file if dynamic generation fails
                print(f"Warning: Dynamic generation failed ({str(e)}), attempting to load from file")
                try:
                    with open(json_file_path) as f:
                        return get_definition_by_model(json.load(f), model)
                except FileNotFoundError:
                    raise ValueError(
                        f"Failed to generate parameters dynamically and no JSON file found at {json_file_path}"
                    )


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""


class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message
