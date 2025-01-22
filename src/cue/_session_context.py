import logging
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SessionContext(BaseModel):
    assistant_id: Optional[str] = None
    conversation_id: Optional[str] = None

    def update(self, **kwargs) -> "SessionContext":
        """Update fields and return self for method chaining"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
