import json
import logging
from typing import Any, Union, Optional
from collections.abc import Iterator, Generator

import tiktoken
from pydantic import BaseModel

from ..schemas import MessageFields

logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize TokenCounter with specified encoding."""
        self.encoding_name = encoding_name
        self._setup_encoding()

    @staticmethod
    def count_token(content: str, model: Optional[str] = None) -> int:
        """Return the rough number of tokens in a string.
        Reference:
        - https://github.com/openai/tiktoken
        - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            if model:
                if "gpt-4o" in model:
                    encoding = tiktoken.get_encoding("o200k_base")
                elif "gpt-4" in model or "text-embedding" in model:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(content))
        except Exception as e:
            logging.error(f"Error getting encoding: {e}")
            return 0

    def _setup_encoding(self) -> None:
        """Initialize the tiktoken encoding."""
        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
        except Exception as e:
            logger.error(f"Error initializing encoding: {e}")
            raise RuntimeError(f"Failed to initialize encoding: {e}")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0

    def safe_serialize(self, obj: Any) -> Any:
        """Safely serialize complex objects to JSON-compatible format."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, BaseModel):
            return self.safe_serialize(obj.model_dump())
        elif isinstance(obj, dict):
            return {k: self.safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.safe_serialize(item) for item in list(obj)]
        elif isinstance(obj, (Generator, Iterator)):
            try:
                return [self.safe_serialize(item) for item in list(obj)]
            except Exception as e:
                logger.warning(f"Could not serialize generator/iterator: {e}")
                return str(obj)
        else:
            return str(obj)

    def contains_image_url_type(self, data: Any) -> bool:
        """Recursively check if data contains image URL type content."""
        if isinstance(data, dict):
            if data.get("type") in ["image", "image_url"]:
                return True
            return any(self.contains_image_url_type(value) for value in data.values())
        elif isinstance(data, (list, tuple, set)):
            return any(self.contains_image_url_type(item) for item in data)
        return False

    def count_message(self, message: Union[BaseModel, dict[str, Any]]) -> int:
        """Count tokens in a message object."""
        try:
            if isinstance(message, BaseModel):
                return self._count_tokens(message.model_dump_json())
            else:
                return self.count_dict_tokens(message)
        except Exception as e:
            logger.error(f"Error counting message tokens: {e}, message type: {type(message)}")
            return 0

    def count_dict_tokens(self, data: Union[dict, Any]) -> int:
        """Count tokens in a TypedDict or dictionary structure."""
        try:
            if self.contains_image_url_type(data):
                logger.debug("Skip to count token for image data")
                return 0

            # First safely serialize the data
            serializable_data = self.safe_serialize(data)

            # Convert to JSON string for token counting
            try:
                json_str = json.dumps(serializable_data)
                return self._count_tokens(json_str)
            except Exception as e:
                logger.error(f"Error in JSON conversion after serialization: {e}")
                # Fallback: try to count tokens in the string representation
                return self._count_tokens(str(serializable_data))

        except Exception as e:
            logger.error(f"Error in token counting: {e}, data type: {type(data)}, data: {data}")
            return 0

    def count_messages_tokens(self, messages: list[Union[BaseModel, dict[str, Any]]]) -> int:
        """
        Get the total token count for all messages, handling BaseModel and Dict separately.

        Args:
            messages: List of messages, each either a BaseModel or Dict

        Returns:
            int: Total number of tokens across all messages

        Raises:
            ValueError: If an unsupported message type is encountered
        """
        total_tokens = 0

        for msg in messages:
            try:
                if isinstance(msg, BaseModel):
                    # Convert BaseModel to dict excluding specific fields
                    msg_dict = msg.model_dump_json(exclude={MessageFields.MSG_ID, MessageFields.MODEL})
                    tokens = self._count_tokens(msg_dict)
                elif isinstance(msg, dict):
                    # For dict messages, exclude fields if they exist
                    msg_copy = msg.copy()
                    msg_copy.pop(MessageFields.MSG_ID, None)
                    msg_copy.pop(MessageFields.MODEL, None)
                    tokens = self.count_dict_tokens(msg_copy)
                else:
                    raise ValueError(f"Unsupported message type: {type(msg)}. Expected BaseModel or dict.")

                total_tokens += tokens

            except Exception as e:
                logger.error(f"Error counting tokens for message: {e}")
                raise

        logger.debug(f"Total tokens counted: {total_tokens}")
        return total_tokens
