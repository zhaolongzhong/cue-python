import json
import logging
from enum import Enum
from typing import Union, Literal, ClassVar, Optional
from pathlib import Path

from .base import BaseTool, ToolError, ToolResult
from ..services import AutomationClient
from ..schemas.automation import AutomationCreate, AutomationUpdate

logger = logging.getLogger(__name__)


class Command(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    READ = "read"
    DELETE = "delete"
    LIST = "list"


class AutomateTool(BaseTool):
    """
    An automation tool that allows to manage automation operations
    """

    name: ClassVar[Literal["automate"]] = "automate"
    _file_history: dict[Path, list[str]]

    def __init__(self, automation_client: Optional[AutomationClient]):
        self._function = self.automate
        self.automation_client = automation_client
        super().__init__()

    async def __call__(
        self,
        *,
        command: Command,
        conversation_id: Optional[str] = None,
        automation_id: Optional[str] = None,
        schedule: Optional[Union[dict, AutomationCreate, AutomationUpdate]] = None,
        skip: Optional[int] = 0,
        limit: Optional[int] = 100,
        **kwargs,
    ):
        if isinstance(schedule, dict):
            if command == Command.CREATE:
                schedule = AutomationCreate(
                    title=schedule.get("title"),
                    prompt=schedule.get("prompt"),
                    schedule=schedule.get("schedule"),
                    conversation_id=conversation_id,
                    is_enabled=schedule.get("is_enabled", True),
                    default_timezone=schedule.get("default_timezone", "UTC"),
                    email_enabled=schedule.get("email_enabled", False),
                )
            elif command == Command.UPDATE:
                schedule = AutomationUpdate(
                    title=schedule.get("title"),
                    prompt=schedule.get("prompt"),
                    schedule=schedule.get("schedule"),
                    is_enabled=schedule.get("is_enabled"),
                    default_timezone=schedule.get("default_timezone"),
                    email_enabled=schedule.get("email_enabled"),
                )
        return await self.automate(
            command=command,
            conversation_id=conversation_id,
            automation_id=automation_id,
            schedule=schedule,
            skip=skip,
            limit=limit,
            **kwargs,
        )

    async def automate(
        self,
        *,
        command: Command,
        conversation_id: Optional[str] = None,
        automation_id: Optional[str] = None,
        schedule: Optional[Union[AutomationCreate, AutomationUpdate]] = None,
        skip: Optional[int] = 0,
        limit: Optional[int] = 100,
        **kwargs,
    ) -> ToolResult:
        """
        Perform automation operations.

        Args:
            command: Operation to perform (create, update, read, delete, list)
            conversation_id: ID for conversation tracking (required for create)
            automation_id: ID for update/delete/read operations
            schedule: Automation configuration for create/update
            skip: Number of items to skip for list operation
            limit: Maximum items to return for list operation

        Examples:
            # Create daily automation
            await tool.automate(
                command=Command.CREATE,
                conversation_id="conv_123",
                schedule=AutomationCreate(
                    title="Daily Report",
                    prompt="Generate daily report",
                    schedule="FREQ=DAILY;INTERVAL=1"
                )
            )
        """
        if self.automation_client is None:
            raise ToolError("Automation service is not enabled")
        # if not conversation_id or :
        conversation_id = self.automation_client.conversation_id

        try:
            if command == Command.CREATE:
                if not conversation_id:
                    raise ToolError("conversation_id is required for create operation")
                self._validate_schedule(schedule)
                return await self.create(conversation_id=conversation_id, schedule=schedule)

            elif command == Command.UPDATE:
                if not automation_id:
                    raise ToolError("automation_id is required for update operation")
                return await self.update(automation_id=automation_id, schedule=schedule)

            elif command == Command.LIST:
                return await self.list(conversation_id=conversation_id, skip=skip, limit=limit)
            elif command == Command.DELETE:
                if not automation_id:
                    raise ToolError("automation_id is required for delete operation")
                return await self.delete(automation_id=automation_id)
            else:
                raise ToolError(f"Unsupported command: {command}")
        except Exception as e:
            raise ToolError(f"Failed to execute automation command: {str(e)}")

    def _validate_schedule(self, schedule: AutomationUpdate) -> None:
        """Validate schedule parameters before making API calls"""
        if not schedule:
            raise ToolError("Schedule is required")
        if not schedule.title:
            raise ToolError("Title is required")
        if not schedule.prompt:
            raise ToolError("Prompt is required")
        if not schedule.schedule:
            raise ToolError("Schedule is required")

    async def create(
        self,
        *,
        conversation_id: str,
        schedule: AutomationCreate,
    ) -> ToolResult:
        """Create a new automation schedule"""
        try:
            if not schedule or not schedule.title or not schedule.prompt or not schedule.schedule:
                raise ToolError("Schedule must include title, prompt, and schedule")

            automation_create = AutomationCreate(
                title=schedule.title,
                prompt=schedule.prompt,
                schedule=schedule.schedule,
                conversation_id=conversation_id,
                is_enabled=schedule.is_enabled if schedule.is_enabled is not None else True,
                default_timezone=schedule.default_timezone or "UTC",
                email_enabled=schedule.email_enabled if schedule.email_enabled is not None else False,
            )

            result = await self.automation_client.create(automation_create)
            return ToolResult(output=f"Automation created successfully with id: {result.id}")
        except Exception as e:
            raise ToolError(f"Failed to create automation: {str(e)}")

    async def update(
        self,
        *,
        automation_id: str,
        schedule: AutomationUpdate,
    ) -> ToolResult:
        """Update an existing automation"""
        try:
            automation_update = AutomationUpdate(
                title=schedule.title,
                prompt=schedule.prompt,
                schedule=schedule.schedule,
                is_enabled=schedule.is_enabled,
                default_timezone=schedule.default_timezone,
                email_enabled=schedule.email_enabled,
            )

            await self.automation_client.update(automation_id=automation_id, automation=automation_update)
            return ToolResult(output=f"Automation {automation_id} updated successfully")
        except Exception as e:
            raise ToolError(f"Failed to update automation {automation_id}: {str(e)}")

    async def read(
        self,
        *,
        automation_id: str,
    ) -> ToolResult:
        """Read an automation's details"""
        try:
            result = await self.automation_client.get(automation_id)
            if not result:
                return ToolResult(output=f"No automation found with id: {automation_id}")

            details = {
                "id": automation_id,
                "title": result.title,
                "prompt": result.prompt,
                "schedule": result.schedule,
                "is_enabled": result.is_enabled,
                "conversation_id": result.conversation_id,
                "default_timezone": result.default_timezone,
                "email_enabled": result.email_enabled,
            }

            return ToolResult(output=json.dumps(details, indent=2))
        except Exception as e:
            raise ToolError(f"Failed to read automation {automation_id}: {str(e)}")

    async def delete(
        self,
        *,
        automation_id: str,
    ) -> ToolResult:
        """Delete an automation"""
        try:
            await self.automation_client.delete(automation_id)
            return ToolResult(output=f"Automation {automation_id} deleted successfully")
        except Exception as e:
            raise ToolError(f"Failed to delete automation {automation_id}: {str(e)}")

    async def list(
        self,
        conversation_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> ToolResult:
        """List all automations"""
        try:
            results = await self.automation_client.list(conversation_id=conversation_id, skip=skip, limit=limit)
            if not results:
                return ToolResult(output="No automations found")

            automations_list = []
            for automation in results:
                auto_info = {
                    "id": automation.id,
                    "title": automation.title,
                    "is_enabled": automation.is_enabled,
                    "conversation_id": automation.conversation_id,
                    "schedule": automation.schedule,
                }
                automations_list.append(auto_info)

            return ToolResult(output=json.dumps(automations_list, indent=2))
        except Exception as e:
            raise ToolError(f"Failed to list automations: {str(e)}")
