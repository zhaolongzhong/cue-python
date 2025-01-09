from datetime import timedelta
from typing import Optional, Union
from pydantic import BaseModel, Field, validator
import croniter
import datetime

class ScheduleConfig(BaseModel):
    """Configuration for scheduled agent tasks.
    
    Either interval or cron should be specified. The scheduled_instruction will be
    sent as a user message to the agent at the scheduled time.
    """
    
    interval: Optional[timedelta] = Field(
        None,
        description="Time interval between runs (e.g., timedelta(hours=1))"
    )
    cron: Optional[str] = Field(
        None,
        description="Cron expression for scheduling (e.g., '0 * * * *' for hourly)"
    )
    scheduled_instruction: str = Field(
        ...,
        description="Instruction/prompt to send to agent. Can use {time} placeholder."
    )
    scheduler_name: Optional[str] = Field(
        "system",
        description="Name to show as the scheduler's identity in messages"
    )
    enabled: bool = Field(
        True,
        description="Whether this schedule is active"
    )
    max_concurrent: int = Field(
        1,
        description="Maximum number of concurrent scheduled runs allowed"
    )
    
    @validator('cron')
    def validate_cron(cls, v):
        if v:
            try:
                # Verify it's a valid cron expression
                croniter.croniter(v, datetime.datetime.now())
            except ValueError as e:
                raise ValueError(f"Invalid cron expression: {e}")
        return v
    
    @validator('interval', 'cron')
    def validate_schedule_type(cls, v, values):
        # Ensure either interval or cron is specified
        if 'interval' in values and values['interval'] is None and 'cron' in values and values['cron'] is None:
            raise ValueError("Either interval or cron must be specified")
        return v
    
    @validator('max_concurrent')
    def validate_max_concurrent(cls, v):
        if v < 1:
            raise ValueError("max_concurrent must be at least 1")
        return v

    def get_next_run_time(self, from_time: Optional[datetime.datetime] = None) -> datetime.datetime:
        """Calculate the next run time based on the schedule configuration."""
        if not from_time:
            from_time = datetime.datetime.now()
            
        if self.interval:
            return from_time + self.interval
        elif self.cron:
            cron = croniter.croniter(self.cron, from_time)
            return cron.get_next(datetime.datetime)
        else:
            raise ValueError("No valid schedule configuration found")