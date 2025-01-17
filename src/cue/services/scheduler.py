"""Scheduler service for managing scheduled tasks."""

import uuid
import asyncio
import logging
import datetime
from typing import Dict, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""

    id: str
    instruction: str
    schedule_time: datetime.datetime
    callback: Callable
    args: tuple = ()
    kwargs: dict = None
    is_completed: bool = False
    completed_at: Optional[datetime.datetime] = None

    def __post_init__(self):
        """Initialize kwargs if None."""
        if self.kwargs is None:
            self.kwargs = {}


class SchedulerService:
    """Service for managing scheduled tasks."""

    _instance = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize scheduler service."""
        if not hasattr(self, "initialized"):
            self.tasks: Dict[str, ScheduledTask] = {}
            self.running = False
            self._task: Optional[asyncio.Task] = None
            self.initialized = True

    def schedule_task(
        self, instruction: str, schedule_time: datetime.datetime, callback: Callable, *args, **kwargs
    ) -> str:
        """Schedule a new task.

        Args:
            instruction: Task instruction/description
            schedule_time: When to execute the task
            callback: Function to call when task is due
            *args: Positional arguments for callback
            **kwargs: Keyword arguments for callback

        Returns:
            str: Task ID
        """
        task_id = str(uuid.uuid4())
        task = ScheduledTask(
            id=task_id,
            instruction=instruction,
            schedule_time=schedule_time,
            callback=callback,
            args=args,
            kwargs=kwargs,
        )
        self.tasks[task_id] = task
        logger.info(f"Scheduled task {task_id} for {schedule_time}")
        return task_id

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Cancelled task {task_id}")
            return True
        return False

    async def _execute_task(self, task: ScheduledTask):
        """Execute a task."""
        try:
            await task.callback(*task.args, **task.kwargs)
            logger.info(f"Completed task {task.id}")
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
        finally:
            task.is_completed = True  # Mark as completed even if it fails
            task.completed_at = datetime.datetime.now()

    async def _check_tasks(self):
        """Check and execute due tasks."""
        while self.running:
            now = datetime.datetime.now()
            # Find tasks that are due
            due_tasks = [task for task in self.tasks.values() if not task.is_completed and task.schedule_time <= now]

            # Execute due tasks and wait for them to complete
            for task in due_tasks:
                await self._execute_task(task)

            # Clean up tasks that have been completed for at least 5 seconds
            self.tasks = {
                tid: task
                for tid, task in self.tasks.items()
                if not (task.is_completed and task.completed_at and (now - task.completed_at).total_seconds() > 5)
            }

            await asyncio.sleep(1)  # Check every second

    async def start(self):
        """Start the scheduler service."""
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._check_tasks())
            logger.info("Scheduler service started")

    async def stop(self):
        """Stop the scheduler service."""
        if self.running:
            self.running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
            logger.info("Scheduler service stopped")
