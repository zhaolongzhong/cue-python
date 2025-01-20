from enum import Enum


class MessageFields(str, Enum):
    """Message core field names."""

    MSG_ID = "msg_id"
    MODEL = "model"
