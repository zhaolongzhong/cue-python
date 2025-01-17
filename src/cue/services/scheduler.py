"""Service for managing scheduled tasks."""
import enum
import asyncio
import inspect
import logging
import datetime
from typing import Callable, Optional

from .task_client import TaskClient
from ..schemas.scheduled_task import TaskType as SchemaTaskType, ScheduledTaskCreate, ScheduledTaskMetadata

logger = logging.getLogger(__name__)


class TaskType(enum.Enum):
    """Task type enumeration."""
    ONE_TIME = "one_time"
    RECURRING = "recurring"


class SchedulerService:
    """Service for managing scheduled tasks."""

    _instance = None

    def __new__(cls, task_client: Optional[TaskClient] = None):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, task_client: Optional[TaskClient] = None):
        """Initialize scheduler service."""
        if not hasattr(self, "initialized"):
            if task_client is None:
                raise ValueError("TaskClient is required")
            self.task_client = task_client
            self.running = False
            self._task: Optional[asyncio.Task] = None
            self.initialized = True

    def _get_callback_info(self, callback: Callable) -> tuple[str, str]:
        """Get module and function name for a callback."""
        module = inspect.getmodule(callback)
        module_name = module.__name__
        function_name = callback.__name__
        return module_name, function_name

    async def schedule_task(
        self,
        instruction: str,
        schedule_time: datetime.datetime,
        callback: Callable,
        *args,
        task_type: TaskType = TaskType.ONE_TIME,
        interval: Optional[datetime.timedelta] = None,
        **kwargs,
    ) -> str:
        """Schedule a new task.

        Args:
            instruction: Task instruction/description
            schedule_time: When to execute the task
            callback: Function to call when task is due
            *args: Positional arguments for callback
            task_type: Type of task (one-time or recurring)
            interval: For recurring tasks, time between executions
            **kwargs: Keyword arguments for callback

        Returns:
            str: Task ID
            
        Raises:
            ValueError: If interval is missing for recurring tasks
        """
        logger.debug(f"Scheduling task: {instruction}")
        logger.debug(f"Task type: {task_type}")
        logger.debug(f"Schedule time: {schedule_time}")
        logger.debug(f"Interval: {interval}")

        if task_type == TaskType.RECURRING and interval is None:
            raise ValueError("Interval is required for recurring tasks")

        module_name, function_name = self._get_callback_info(callback)
        logger.debug(f"Callback: {module_name}.{function_name}")

        # Make sure schedule_time has no timezone to match NOW comparisons
        if schedule_time.tzinfo is not None:
            schedule_time = schedule_time.replace(tzinfo=None)

        metadata = ScheduledTaskMetadata(
            callback_module=module_name,
            callback_name=function_name,
            callback_args=list(args),
            callback_kwargs=kwargs
        )

        task = ScheduledTaskCreate(
            instruction=instruction,
            schedule_time=schedule_time,
            task_type=SchemaTaskType(task_type.value),
            interval=interval,
            metadata=metadata
        )

        result = await self.task_client.create(task)
        if not result:
            raise RuntimeError("Failed to create task")

        return result.id

    async def _execute_task(self, task_id: str) -> None:
        """Execute a task."""
        task = await self.task_client.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        if task.is_completed:
            logger.debug(f"Task {task_id} already completed")
            return

        logger.debug(f"Executing task {task.id}: {task.instruction}")
        logger.debug(f"Task details: schedule_time={task.schedule_time}, interval={task.interval}")
        logger.debug(f"Task callback: {task.metadata.callback_module}.{task.metadata.callback_name}")

        error = None
        try:
            callback = self.task_client.get_callback(task)
            logger.debug(f"Got callback for task {task.id}: {callback}")
            await callback(*task.metadata.callback_args, **task.metadata.callback_kwargs)
            logger.info(f"Completed task {task.id}")
            await self.task_client.mark_completed(task.id)
        except Exception as e:
            import traceback
            error = str(e)
            logger.error(f"Error executing task {task.id}: {e}\n{traceback.format_exc()}")
            await self.task_client.mark_completed(task.id, error)

        # Reschedule if it's a recurring task
        if task.interval:
            next_time = datetime.datetime.now() + task.interval
            logger.debug(f"Rescheduling recurring task {task.id} for {next_time}")
            await self.task_client.reschedule_recurring(task.id, next_time)

    async def _check_tasks(self) -> None:
        """Check and execute due tasks."""
        while self.running:
            try:
                logger.debug("Checking for due tasks...")
                due_tasks = await self.task_client.get_due_tasks()
                logger.debug(f"Found {len(due_tasks)} due tasks")
                for task in due_tasks:
                    logger.debug(f"Processing task {task.id} ({task.instruction})")
                    await self._execute_task(task.id)
            except Exception as e:
                import traceback
                logger.error(f"Error checking tasks: {e}\n{traceback.format_exc()}")

            logger.debug("Sleeping for 1 second")
            await asyncio.sleep(1)  # Check every second

    async def start(self) -> None:
        """Start the scheduler service."""
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._check_tasks())
            logger.info("Scheduler service started")

    async def stop(self) -> None:
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
