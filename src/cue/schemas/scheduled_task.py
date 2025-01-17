"""Scheduled task schemas."""
import enum
import datetime
from typing import Any, Dict, Optional

from pydantic import Field, BaseModel


class TaskType(str, enum.Enum):
    """Task type enumeration."""
    ONE_TIME = "one_time"
    RECURRING = "recurring"


class ScheduledTaskMetadata(BaseModel):
    """Scheduled task metadata."""
    callback_module: str = Field(description="Python module containing the callback")
    callback_name: str = Field(description="Name of the callback function")
    callback_args: Optional[list] = Field(default_factory=list, description="Positional arguments for callback")
    callback_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Keyword arguments for callback")


class ScheduledTask(BaseModel):
    """Scheduled task model."""
    id: str = Field(description="Unique identifier for the task")
    instruction: str = Field(description="Task instruction/description")
    schedule_time: datetime.datetime = Field(description="When to execute the task")
    task_type: TaskType = Field(default=TaskType.ONE_TIME, description="Type of task (one-time or recurring)")
    interval: Optional[datetime.timedelta] = Field(default=None, description="For recurring tasks, time between executions")
    metadata: ScheduledTaskMetadata = Field(description="Task metadata including callback information")
    is_completed: bool = Field(default=False, description="Whether the task has been completed")
    completed_at: Optional[datetime.datetime] = Field(default=None, description="When the task was completed")
    error: Optional[str] = Field(default=None, description="Error message if task failed")


class ScheduledTaskCreate(BaseModel):
    """Task creation model."""
    instruction: str = Field(description="Task instruction/description")
    schedule_time: datetime.datetime = Field(description="When to execute the task")
    task_type: TaskType = Field(default=TaskType.ONE_TIME, description="Type of task (one-time or recurring)")
    interval: Optional[datetime.timedelta] = Field(default=None, description="For recurring tasks, time between executions")
    metadata: ScheduledTaskMetadata = Field(description="Task metadata including callback information")


class ScheduledTaskUpdate(BaseModel):
    """Task update model."""
    instruction: Optional[str] = Field(default=None, description="Task instruction/description")
    schedule_time: Optional[datetime.datetime] = Field(default=None, description="When to execute the task")
    interval: Optional[datetime.timedelta] = Field(default=None, description="For recurring tasks, time between executions")
    metadata: Optional[ScheduledTaskMetadata] = Field(default=None, description="Task metadata including callback information")
    is_completed: Optional[bool] = Field(default=None, description="Whether the task has been completed")
    completed_at: Optional[datetime.datetime] = Field(default=None, description="When the task was completed")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
