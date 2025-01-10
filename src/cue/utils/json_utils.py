import os
import json
import logging
import dataclasses
from typing import Any, Optional
from pathlib import Path

import blobfile as bf
import pydantic

logger = logging.getLogger(__name__)


def get_jsonl(path: str) -> list[dict]:
    """
    Reference: https://github.com/openai/evals/blob/main/evals/data.py

    Extract json lines from the given path.
    If the path is a directory, look in subpaths recursively.

    Return all lines from all jsonl files as a single list.
    """
    if bf.isdir(path):
        result = []
        for filename in bf.listdir(path):
            if filename.endswith(".jsonl"):
                result += get_jsonl(os.path.join(path, filename))
        return result
    return _get_jsonl_file(path)


def _get_jsonl_file(path):
    logger.info("Fetching %s", path)
    with open(path, encoding="utf-8") as f:
        return [_decode_json(line, path, i + 1) for i, line in enumerate(f)]


def _decode_json(line, path, line_number):
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        custom_error_message = (
            f"Error parsing JSON on line {line_number}: {e.msg} at {path}:{line_number}:{e.colno}, line: {line}"
        )
        logger.error(custom_error_message)
        raise ValueError(custom_error_message) from None


def append_jsonl(data: Any, file_path: str, ensure_ascii: bool = False, **kwargs: Any) -> None:
    """
    Append a JSON object to a JSONL file.

    :param data: The data to be serialized and appended.
    :param file_path: Path to the JSONL file.
    :param ensure_ascii: If True, ensure that non-ASCII characters are escaped.
    :param kwargs: Additional keyword arguments to pass to the jsondumps function.
    """
    # Serialize the data to a JSON formatted string using jsondumps
    json_line = jsondumps(data, ensure_ascii=ensure_ascii, **kwargs)

    # Append the JSON line to the file with a newline
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json_line + "\n")


def _to_py_types(o: Any, exclude_keys: list[str]) -> Any:
    if isinstance(o, dict):
        return {k: _to_py_types(v, exclude_keys=exclude_keys) for k, v in o.items() if k not in exclude_keys}

    if isinstance(o, list):
        return [_to_py_types(v, exclude_keys=exclude_keys) for v in o]

    if isinstance(o, Path):
        return o.as_posix()

    if dataclasses.is_dataclass(o):
        return _to_py_types(dataclasses.asdict(o), exclude_keys=exclude_keys)

    # pydantic data classes
    if isinstance(o, pydantic.BaseModel):
        return {
            k: _to_py_types(v, exclude_keys=exclude_keys)
            for k, v in json.loads(o.json()).items()
            if k not in exclude_keys
        }

    return o


class EnhancedJSONEncoder(json.JSONEncoder):
    def __init__(self, exclude_keys: Optional[list[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.exclude_keys = exclude_keys if exclude_keys else []

    def default(self, o: Any) -> str:
        return _to_py_types(o, self.exclude_keys)


def jsondumps(o: Any, ensure_ascii: bool = False, **kwargs: Any) -> str:
    # The JSONEncoder class's .default method is only applied to dictionary values,
    # not keys. In order to exclude keys from the output of this jsondumps method
    # we need to exclude them outside the encoder.
    if isinstance(o, dict) and "exclude_keys" in kwargs:
        for key in kwargs["exclude_keys"]:
            del o[key]
    return json.dumps(o, cls=EnhancedJSONEncoder, ensure_ascii=ensure_ascii, **kwargs)
