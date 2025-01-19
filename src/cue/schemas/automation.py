from typing import Optional

from pydantic import BaseModel


class Automation(BaseModel):
    id: Optional[str] = None
    title: str
    prompt: str
    is_enabled: bool = True
    conversation_id: str
    schedule: str  # iCal format string
    default_timezone: str = "UTC"
    email_enabled: bool = False


class AutomationCreate(Automation):
    pass


class AutomationUpdate(BaseModel):
    title: Optional[str] = None
    prompt: Optional[str] = None
    schedule: Optional[str] = None
    is_enabled: Optional[bool] = None
    default_timezone: Optional[str] = None
    email_enabled: Optional[bool] = None
