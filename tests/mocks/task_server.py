"""Mock task server for testing."""
import uuid
from typing import Dict, List, Optional
from datetime import datetime

from cue.schemas.scheduled_task import ScheduledTask, ScheduledTaskCreate, ScheduledTaskUpdate


class MockTaskServer:
    """Mock task server."""

    def __init__(self):
        """Initialize mock server."""
        self.tasks: Dict[str, ScheduledTask] = {}

    async def create_task(self, task: ScheduledTaskCreate) -> ScheduledTask:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        task_dict = task.model_dump()
        task_dict["id"] = task_id
        new_task = ScheduledTask(**task_dict)
        self.tasks[task_id] = new_task
        return new_task

    async def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    async def list_tasks(self, skip: int = 0, limit: int = 100) -> List[ScheduledTask]:
        """List tasks."""
        tasks = list(self.tasks.values())
        return tasks[skip:skip + limit]

    async def update_task(self, task_id: str, task: ScheduledTaskUpdate) -> Optional[ScheduledTask]:
        """Update a task."""
        if task_id not in self.tasks:
            return None

        current_task = self.tasks[task_id]
        update_dict = task.model_dump(exclude_unset=True)
        updated_task = ScheduledTask(
            **{**current_task.model_dump(), **update_dict}
        )
        self.tasks[task_id] = updated_task
        return updated_task

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False

    async def get_due_tasks(self, before: Optional[datetime] = None) -> List[ScheduledTask]:
        """Get tasks due for execution."""
        before = before or datetime.now()
        return [
            task for task in self.tasks.values()
            if not task.is_completed and task.schedule_time <= before
        ]
