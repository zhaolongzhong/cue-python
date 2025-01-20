from pydantic import BaseModel

__all__ = ["RunUsage"]


class RunUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0
