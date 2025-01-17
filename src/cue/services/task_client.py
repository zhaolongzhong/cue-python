"""Client for task-related operations."""
import logging
import importlib
from typing import Any, List, Callable, Optional
from datetime import datetime

from .transport import HTTPTransport, ResourceClient, WebSocketTransport
from ..schemas.scheduled_task import ScheduledTask, ScheduledTaskCreate, ScheduledTaskUpdate

logger = logging.getLogger(__name__)


def import_callback(module_name: str, function_name: str) -> Callable:
    """Import a callback function from a module."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {function_name} from {module_name}: {e}")


class TaskClient(ResourceClient):
    """Client for task-related operations."""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        """Initialize task client."""
        super().__init__(http, ws)

    async def create(self, task: ScheduledTaskCreate) -> Optional[ScheduledTask]:
        """Create a new task."""
        response = await self._http.request("POST", "/tasks", data=task.model_dump())
        if not response:
            logger.error("Create task failed")
            return None
        return ScheduledTask(**response)

    async def get(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        response = await self._http.request("GET", f"/tasks/{task_id}")
        if not response:
            return None
        return ScheduledTask(**response)

    async def list(self, skip: int = 0, limit: int = 100) -> List[ScheduledTask]:
        """List tasks."""
        response = await self._http.request(
            "GET",
            "/tasks",
            params={"skip": skip, "limit": limit}
        )
        return [ScheduledTask(**task) for task in response]

    async def update(self, task_id: str, task: ScheduledTaskUpdate) -> ScheduledTask:
        """Update a task."""
        response = await self._http.request(
            "PUT",
            f"/tasks/{task_id}",
            data=task.model_dump(exclude_unset=True)
        )
        return ScheduledTask(**response)

    async def delete(self, task_id: str) -> None:
        """Delete a task."""
        await self._http.request("DELETE", f"/tasks/{task_id}")

    async def get_due_tasks(
        self,
        before: Optional[datetime] = None
    ) -> List[ScheduledTask]:
        """Get tasks due for execution.
        
        Args:
            before: Get tasks scheduled before this time. Defaults to now.
        """
        before = before or datetime.now()
        # Strip timezone info to match scheduler's naive datetimes
        if before.tzinfo is not None:
            before = before.replace(tzinfo=None)
        response = await self._http.request(
            "GET",
            "/tasks/due",
            params={"before": before.isoformat()}
        )
        if not response:
            return []
        return [ScheduledTask(**task) for task in response]

    def get_callback(self, task: ScheduledTask) -> Callable[..., Any]:
        """Get the callback function for a task."""
        metadata = task.metadata
        return import_callback(metadata.callback_module, metadata.callback_name)

    async def mark_completed(
        self,
        task_id: str,
        error: Optional[str] = None
    ) -> Optional[ScheduledTask]:
        """Mark a task as completed."""
        update = ScheduledTaskUpdate(
            is_completed=True,
            completed_at=datetime.now(),
            error=error
        )
        try:
            return await self.update(task_id, update)
        except Exception as e:
            logger.error(f"Failed to mark task {task_id} as completed: {e}")
            return None

    async def reschedule_recurring(
        self,
        task_id: str,
        next_time: datetime
    ) -> Optional[ScheduledTask]:
        """Reschedule a recurring task."""
        # Make sure next_time has no timezone to match scheduler's naive datetimes
        if next_time.tzinfo is not None:
            next_time = next_time.replace(tzinfo=None)

        update = ScheduledTaskUpdate(
            schedule_time=next_time,
            is_completed=False,
            completed_at=None,
            error=None
        )
        try:
            return await self.update(task_id, update)
        except Exception as e:
            logger.error(f"Failed to reschedule task {task_id}: {e}")
            return None
