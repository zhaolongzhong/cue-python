from typing import Literal

from pydantic import BaseModel

__all__ = ["Author"]


class Author(BaseModel):
    name: str | None = None
    role: Literal["user", "assistant", "tool", "system"]
    metadata: dict | None = None
