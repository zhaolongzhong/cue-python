from typing import Optional
from datetime import datetime

from pydantic import BaseModel, field_serializer

from .run_usage import RunUsage

__all__ = ["RunUsageAndLimits"]


class RunUsageAndLimits(BaseModel):
    usage: RunUsage  # current usage
    usage_limits: RunUsage  # max usage
    start_time: datetime = datetime.now()
    end_time: Optional[datetime] = None
    duration: Optional[int] = None

    @field_serializer("start_time")
    def serialize_start_time(self, v: datetime) -> str:
        return v.isoformat()

    @field_serializer("end_time")
    def serialize_end_time(self, v: datetime) -> str:
        return v.isoformat()
