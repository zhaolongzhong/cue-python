from enum import Enum

__all__ = ["Role"]


class Role(str, Enum):
    user = "user"
    tool = "tool"
    system = "system"
    assistant = "assistant"
