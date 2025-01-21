import os
import json
import logging
from typing import Any, Union, Optional
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel

from ..config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class DebugUtils:
    @staticmethod
    def debug_print_messages(messages: list[dict[str, Any]], indent: int = 2, tag: Optional[str] = None) -> None:
        """
        Pretty prints a list of message dictionaries.

        Args:
        messages (List[Dict[str, Any]]): The list of message dictionaries to print.
        indent (int): The number of spaces to use for indentation. Default is 2.
        """
        try:
            size = len(messages)
            for i, message in enumerate(messages, 1):
                if isinstance(message, BaseModel):
                    # message = message.model_dump()
                    pass
                if isinstance(message, dict) or isinstance(message, list):
                    processed_message = truncate_image_data(message)
                    logger.debug(f"{tag} Message {i}/{size}: {json.dumps(processed_message, indent=indent + 2)}")
                else:
                    logger.debug(f"{tag} Message {i}/{size}: {message}")

        except Exception as e:
            # when it's image, Object of type SerializationIterator is not JSON serializable
            if "SerializationIterator" not in str(e):
                logger.error(f"debug_print error: {e}")
            else:
                logger.debug("debug_print skip print last message, it might be image")

    @staticmethod
    def _serialize_message(msg: Union[dict, BaseModel]) -> dict[str, Any]:
        """
        Convert message to a serializable dictionary
        """
        if isinstance(msg, BaseModel):
            return msg.model_dump()
        return msg

    @staticmethod
    def take_snapshot(
        messages: list[Union[dict, BaseModel]],
        pretty: bool = True,
        subfolder: Optional[str] = None,
        suffix: Optional[str] = None,
        with_timestamp: Optional[bool] = False,
    ):
        """
        Record current message list to a file in JSONL format

        Args:
            messages: List of messages. Each message can be:
                - Dictionary
                - Pydantic BaseModel
            pretty: If True, format the JSON with indentation for better readability
            suffix: If provided, append it at the end otherwise use timestamp
            with_timestamp: If provided, append timestamp
            subfolder: If provided, create a subfolder

        Returns:
            Path: Path to the created snapshot file
        """
        if not messages:
            return

        if suffix:
            if with_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = f"{suffix}_{timestamp}"
            filename_suffix = suffix
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_suffix = timestamp

        base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        if subfolder:
            snapshot_base_path = base_dir / "logs/snapshot" / subfolder
        else:
            snapshot_base_path = base_dir / "logs/snapshot"

        if pretty:
            file_name = f"snapshot_{filename_suffix}_readable.json"
            snapshot_path = snapshot_base_path / file_name
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            serialized_messages = [DebugUtils._serialize_message(msg) for msg in messages]

            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(serialized_messages, f, ensure_ascii=False, indent=2)

        file_name = f"snapshot_{filename_suffix}.jsonl"
        snapshot_path = snapshot_base_path / file_name
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        with open(snapshot_path, "w", encoding="utf-8") as f:
            for msg in messages:
                serialized_msg = DebugUtils._serialize_message(msg)
                json.dump(serialized_msg, f, ensure_ascii=False)
                f.write("\n")

        return snapshot_path

    def read_snapshot(self, file_path: Path):
        """
        Read and display contents of a snapshot file
        """
        if file_path.suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    print(json.loads(line))
        else:
            with open(file_path, encoding="utf-8") as f:
                print(json.load(f))

    @staticmethod
    def log_chat(msg: dict, _tag: Optional[str] = None) -> None:
        """Save user and assistant message to a file"""
        if settings.is_production:
            return
        base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        user_input_path = base_dir / "logs/chat.jsonl"
        user_input_path.parent.mkdir(parents=True, exist_ok=True)
        msg_with_timestamp = {"timestamp": datetime.now().isoformat(), **msg}
        with open(user_input_path, "a", encoding="utf-8") as f:
            json.dump(msg_with_timestamp, f)
            f.write("\n")


def truncate_image_data(obj, max_length=50):
    """Recursively process dictionary/list to truncate image data"""
    if isinstance(obj, dict):
        # Case 1: Handle openai image_url structure
        if obj.get("type") == "image_url" and "image_url" in obj:
            image_url = obj["image_url"]
            if "url" in image_url and image_url["url"].startswith("data:image"):
                new_obj = obj.copy()
                base64_part = image_url["url"].split("base64,")[1]
                new_obj["image_url"] = {"url": f"<base64_image_data truncated, length: {len(base64_part)}>"}
                return new_obj
        # Case 2: Handle anthropic image with source structure
        if obj.get("type") == "image" and "source" in obj:
            source = obj["source"]
            if source.get("type") == "base64" and "data" in source:
                new_obj = obj.copy()
                new_obj["source"] = {**source, "data": f"<base64_image_data truncated, length: {len(source['data'])}>"}
                return new_obj
        return {key: truncate_image_data(value, max_length) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [truncate_image_data(item, max_length) for item in obj]
    else:
        return obj
