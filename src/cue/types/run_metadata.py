from uuid import uuid4
from typing import Optional

from pydantic import Field, BaseModel

__all__ = ["RunMetadata"]


class RunMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))  # unique id for identify runner
    runner_name: Optional[str] = "default"
    mode: Optional[str] = "cli"
    last_user_message: str = ""
    user_messages: list[str] = []
    max_turns: int = 30  # maximum turns per run
    current_turn: int = 0
    enable_turn_debug: bool = False  # enable debug for each turn
    token_stats: dict = {}
    metrics: dict = {}
