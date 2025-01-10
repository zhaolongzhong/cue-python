from typing import Any, Union, TypeVar, Optional
from logging import getLogger
from dataclasses import dataclass

logger = getLogger(__name__)

T = TypeVar("T")


@dataclass
class ValidationResult:
    """
    Holds the result of parameter validation.
    """

    value: Any
    is_valid: bool
    error_message: Optional[str] = None


class ParameterValidator:
    """
    Utility class for parameter validation and safe type conversion.
    """

    @staticmethod
    def safe_int(
        value: Any, default: Optional[int] = None, min_value: Optional[int] = None, max_value: Optional[int] = None
    ) -> ValidationResult:
        """
        Safely converts a value to integer with bounds checking.

        Args:
            value: Value to convert
            default: Default value if conversion fails
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            ValidationResult containing the converted value or default
        """
        try:
            if value is None:
                return ValidationResult(default, True)

            if isinstance(value, str) and value.lower() == "none":
                return ValidationResult(default, True)

            converted = int(value)

            if min_value is not None and converted < min_value:
                return ValidationResult(min_value, False, f"Value {converted} is below minimum {min_value}")

            if max_value is not None and converted > max_value:
                return ValidationResult(max_value, False, f"Value {converted} is above maximum {max_value}")

            return ValidationResult(converted, True)

        except (ValueError, TypeError) as e:
            logger.debug(f"Parameter conversion failed: {str(e)}")
            return ValidationResult(default, False, f"Invalid value: {value}")

    @staticmethod
    def safe_string_list(
        value: Optional[Union[str, list[str]]], separator: str = ",", strip: bool = True
    ) -> ValidationResult:
        """
        Safely converts string or list input to list of strings.

        Args:
            value: String or list to process
            separator: Separator to split string if needed
            strip: Whether to strip whitespace from results

        Returns:
            ValidationResult containing the processed list
        """
        try:
            if value is None:
                return ValidationResult([], True)

            if isinstance(value, list):
                result = [str(item).strip() if strip else str(item) for item in value]
                return ValidationResult(result, True)

            if isinstance(value, str):
                if separator in value:
                    result = [item.strip() if strip else item for item in value.split(separator)]
                    return ValidationResult(result, True)
                return ValidationResult([value.strip() if strip else value], True)

            return ValidationResult([], False, f"Invalid value type: {type(value)}")

        except Exception as e:
            logger.debug(f"String list conversion failed: {str(e)}")
            return ValidationResult([], False, f"Processing failed: {str(e)}")

    @staticmethod
    def log_validation_error(result: ValidationResult, param_name: str) -> None:
        """
        Logs validation errors if present.

        Args:
            result: ValidationResult to check
            param_name: Name of parameter for logging
        """
        if not result.is_valid and result.error_message:
            logger.warning(
                f"Parameter validation failed for {param_name}: {result.error_message}. Using value: {result.value}"
            )
