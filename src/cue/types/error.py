from typing import Optional

from pydantic import BaseModel

__all__ = ["ErrorResponse"]


class ErrorResponse(BaseModel):
    message: str
    code: Optional[str] = None
