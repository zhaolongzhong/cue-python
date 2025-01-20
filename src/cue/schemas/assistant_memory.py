from typing import Optional
from datetime import datetime

from pydantic import Field, BaseModel


class AssistantMemoryBase(BaseModel):
    content: str
    metadata: Optional[dict] = None


class AssistantMemoryCreate(AssistantMemoryBase):
    assistant_id: Optional[str] = None


class AssistantMemoryUpdate(BaseModel):
    content: Optional[str] = None
    metadata: Optional[dict] = None


class AssistantMemoryInDBBase(AssistantMemoryBase):
    id: str
    assistant_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class ConfigDict:
        from_attributes = True


class AssistantMemory(AssistantMemoryInDBBase):
    pass


class AssistantMemoryInDB(AssistantMemoryInDBBase):
    pass


class RelevantMemory(AssistantMemoryInDB):
    similarity_score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score between memory and query")


class RelevantMemoriesResponse(BaseModel):
    memories: list[RelevantMemory]
    total_count: int
    threshold_used: float = Field(default=0.05, description="Similarity threshold used for filtering memories")
