from pydantic import BaseModel


class ConversationContext(BaseModel):
    participants: list[str]
