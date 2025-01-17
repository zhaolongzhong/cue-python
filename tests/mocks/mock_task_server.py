"""Mock task server for testing."""
import uuid
from typing import Dict, List, Optional
from datetime import datetime

from cue.schemas.scheduled_task import ScheduledTask, ScheduledTaskCreate, ScheduledTaskUpdate


class MockTaskServer:
    """Mock server for task operations."""

    def __init__(self):
        """Initialize mock server."""
        self.tasks: Dict[str, ScheduledTask] = {}

    async def create_task(self, task: ScheduledTaskCreate) -> ScheduledTask:
        """Create a new task."""
        import logging
        logger = logging.getLogger(__name__)

        task_id = str(uuid.uuid4())
        task_data = task.model_dump()
        task_data['id'] = task_id

        # Ensure schedule_time has no timezone to match NOW comparisons
        if task_data['schedule_time'].tzinfo is not None:
            task_data['schedule_time'] = task_data['schedule_time'].replace(tzinfo=None)

        new_task = ScheduledTask(**task_data)
        self.tasks[task_id] = new_task

        logger.debug(f"Created task {task_id}: {new_task.instruction} "
                    f"(scheduled: {new_task.schedule_time}, "
                    f"interval: {new_task.interval})")

        return new_task

    async def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    async def list_tasks(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[ScheduledTask]:
        """List tasks."""
        tasks = list(self.tasks.values())
        return tasks[skip:skip + limit]

    async def update_task(
        self,
        task_id: str,
        task: ScheduledTaskUpdate
    ) -> Optional[ScheduledTask]:
        """Update a task."""
        if task_id not in self.tasks:
            return None

        current = self.tasks[task_id]
        updates = task.model_dump(exclude_unset=True)
        updated_data = current.model_dump()
        updated_data.update(updates)

        updated_task = ScheduledTask(**updated_data)
        self.tasks[task_id] = updated_task
        return updated_task

    async def delete_task(self, task_id: str) -> None:
        """Delete a task."""
        self.tasks.pop(task_id, None)

    async def get_due_tasks(
        self,
        before: Optional[datetime] = None
    ) -> List[ScheduledTask]:
        """Get tasks due for execution."""
        import logging
        logger = logging.getLogger(__name__)

        before = before or datetime.now()
        logger.debug(f"Getting tasks due before {before} (tzinfo={before.tzinfo})")

        due_tasks = [
            task for task in self.tasks.values()
            if not task.is_completed and task.schedule_time <= before
        ]

        logger.debug(f"Found {len(due_tasks)} due tasks")
        for task in due_tasks:
            logger.debug(f"Due task: {task.id} - {task.instruction} "
                      f"(scheduled: {task.schedule_time}, completed: {task.is_completed})")

        return due_tasks
