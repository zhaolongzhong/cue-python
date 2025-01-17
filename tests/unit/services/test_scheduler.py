"""Tests for scheduler service."""

import asyncio
import datetime
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio

from cue.services.scheduler import TaskType, SchedulerService
from cue.services.transport import HTTPTransport
from cue.services.task_client import TaskClient
from cue.schemas.scheduled_task import (
    TaskType as SchemaTaskType,
    ScheduledTask,
)


class MockHTTPTransport(HTTPTransport):
    """Mock HTTP transport for testing."""

    def __init__(self):
        """Initialize mock transport."""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.next_id = 1

    async def request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Handle HTTP request."""
        if method == "POST" and path == "/tasks":
            # Create task
            task_id = str(self.next_id)
            self.next_id += 1
            task_dict = data or {}
            task_dict["id"] = task_id
            task = ScheduledTask(**task_dict)
            self.tasks[task_id] = task
            return task.model_dump()

        elif method == "GET" and path.startswith("/tasks/"):
            # Get single task
            task_id = path.split("/")[-1]
            task = self.tasks.get(task_id)
            if not task:
                return None
            return task.model_dump()

        elif method == "GET" and path == "/tasks/due":
            # Get due tasks
            now = datetime.datetime.now()
            if params and "before" in params:
                now = datetime.datetime.fromisoformat(params["before"])
            return [
                task.model_dump()
                for task in self.tasks.values()
                if not task.is_completed and task.schedule_time <= now
            ]

        elif method == "PUT" and path.startswith("/tasks/"):
            # Update task
            task_id = path.split("/")[-1]
            task = self.tasks.get(task_id)
            if not task:
                return None
            task_dict = task.model_dump()
            task_dict.update(data or {})
            updated_task = ScheduledTask(**task_dict)
            self.tasks[task_id] = updated_task
            return updated_task.model_dump()

        elif method == "DELETE" and path.startswith("/tasks/"):
            # Delete task
            task_id = path.split("/")[-1]
            if task_id in self.tasks:
                del self.tasks[task_id]
                return {}
            return None

        return None


@pytest.fixture
def http_transport():
    """Create mock HTTP transport."""
    return MockHTTPTransport()


@pytest.fixture
def task_client(http_transport):
    """Create mock task client."""
    return TaskClient(http_transport)


@pytest.fixture
def scheduler(task_client):
    """Create scheduler fixture."""
    return SchedulerService(task_client)


@pytest_asyncio.fixture
async def running_scheduler(scheduler):
    """Create running scheduler fixture."""
    await scheduler.start()
    try:
        yield scheduler
    finally:
        await scheduler.stop()


async def mock_callback(*args, **kwargs):
    """Mock callback for testing."""
    pass


def test_scheduler_singleton(task_client):
    """Test scheduler is singleton."""
    s1 = SchedulerService(task_client)
    s2 = SchedulerService(task_client)
    assert s1 is s2
    assert s1._task_client is s2._task_client


@pytest.mark.asyncio
async def test_one_time_task(scheduler, task_client):
    """Test scheduling a one-time task."""
    schedule_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
    task_id = await scheduler.schedule_task("Test task", schedule_time, mock_callback, task_type=TaskType.ONE_TIME)

    task = await task_client.get(task_id)
    assert task is not None
    assert task.instruction == "Test task"
    assert task.schedule_time == schedule_time.replace(tzinfo=None)  # Compare without timezone
    assert task.task_type == SchemaTaskType.ONE_TIME
    assert task.interval is None  # One-time tasks have no interval
    assert not task.is_completed
    assert task.error is None
    assert task.metadata.callback_module == mock_callback.__module__
    assert task.metadata.callback_name == mock_callback.__name__


@pytest.mark.asyncio
async def test_recurring_task(running_scheduler, task_client):
    """Test recurring task execution."""
    results = []

    async def test_callback():
        results.append(datetime.datetime.now())

    # Schedule recurring task every second
    now = datetime.datetime.now()
    task_id = await running_scheduler.schedule_task(
        "Recurring test", now, test_callback, task_type=TaskType.RECURRING, interval=datetime.timedelta(seconds=1)
    )

    # Wait for multiple executions
    await asyncio.sleep(3.5)  # Should see ~3 executions
    assert len(results) >= 3

    # Verify task state
    task = await task_client.get(task_id)
    assert task.task_type == SchemaTaskType.RECURRING
    assert task.interval == datetime.timedelta(seconds=1)
    assert not task.is_completed  # Recurring tasks should never be marked completed
    assert task.error is None  # No errors should be present

    # Verify intervals between executions
    for i in range(1, len(results)):
        interval = results[i] - results[i - 1]
        assert 0.8 <= interval.total_seconds() <= 1.2  # Allow 200ms variance

    # Verify next schedule time is in the future
    now = datetime.datetime.now()
    assert task.schedule_time > now


@pytest.mark.asyncio
async def test_recurring_task_validation(scheduler, task_client):
    """Test recurring task validation."""
    now = datetime.datetime.now()

    # Test missing interval
    with pytest.raises(ValueError, match="Interval is required for recurring tasks"):
        await scheduler.schedule_task(
            "Invalid recurring task",
            now,
            mock_callback,
            task_type=TaskType.RECURRING  # Missing interval
        )

    # Test one-time task with interval (should be ok but ignored)
    task_id = await scheduler.schedule_task(
        "One-time with interval",
        now,
        mock_callback,
        task_type=TaskType.ONE_TIME,
        interval=datetime.timedelta(seconds=1)
    )

    task = await task_client.get(task_id)
    assert task.task_type == SchemaTaskType.ONE_TIME
    assert task.interval is None  # Interval should be ignored for one-time tasks


@pytest.mark.asyncio
async def test_cancel_task(scheduler, task_client):
    """Test cancelling a task."""
    schedule_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
    task_id = await scheduler.schedule_task("Test task", schedule_time, mock_callback)

    # Get task before cancellation
    task = await task_client.get(task_id)
    assert task is not None
    assert task.task_type == SchemaTaskType.ONE_TIME

    # Cancel and verify
    assert await scheduler.cancel_task(task_id)
    assert await task_client.get(task_id) is None


@pytest.mark.asyncio
async def test_task_execution(running_scheduler, task_client):
    """Test task execution."""
    executed = False
    completed_at = None

    async def test_callback():
        nonlocal executed, completed_at
        executed = True
        completed_at = datetime.datetime.now()

    # Schedule task for immediate execution
    schedule_time = datetime.datetime.now()
    task_id = await running_scheduler.schedule_task("Test execution", schedule_time, test_callback)

    # Verify initial task state
    task = await task_client.get(task_id)
    assert task.task_type == SchemaTaskType.ONE_TIME
    assert not task.is_completed
    assert task.error is None

    # Wait for execution
    await asyncio.sleep(2)
    assert executed

    # Verify task state after execution
    task = await task_client.get(task_id)
    assert task.is_completed
    assert task.completed_at is not None
    assert task.error is None

    # Verify completion time is after schedule time
    assert task.completed_at > schedule_time
    assert task.completed_at >= completed_at


@pytest.mark.asyncio
async def test_multiple_tasks(running_scheduler, task_client):
    """Test handling multiple tasks."""
    results = []
    completed_times = {}

    async def callback(value):
        completed_times[value] = datetime.datetime.now()
        results.append(value)

    now = datetime.datetime.now()
    schedule_time1 = now + datetime.timedelta(milliseconds=100)
    schedule_time2 = now + datetime.timedelta(milliseconds=200)

    # Schedule multiple tasks
    task1_id = await running_scheduler.schedule_task(
        "Task 1",
        schedule_time1,
        callback,
        1,
        task_type=TaskType.ONE_TIME
    )
    task2_id = await running_scheduler.schedule_task(
        "Task 2",
        schedule_time2,
        callback,
        2,
        task_type=TaskType.ONE_TIME
    )

    # Verify initial task states
    task1 = await task_client.get(task1_id)
    task2 = await task_client.get(task2_id)
    assert task1.task_type == SchemaTaskType.ONE_TIME
    assert task2.task_type == SchemaTaskType.ONE_TIME
    assert not task1.is_completed and not task2.is_completed
    assert task1.error is None and task2.error is None

    # Wait for execution
    await asyncio.sleep(1)
    assert results == [1, 2]  # Tasks executed in order

    # Verify final task states
    task1 = await task_client.get(task1_id)
    task2 = await task_client.get(task2_id)

    # Both tasks completed successfully
    assert task1.is_completed and task2.is_completed
    assert task1.completed_at is not None and task2.completed_at is not None
    assert task1.error is None and task2.error is None

    # Tasks executed in correct order
    assert task1.completed_at < task2.completed_at

    # Completion times are after schedule times
    assert task1.completed_at >= schedule_time1.replace(tzinfo=None)
    assert task2.completed_at >= schedule_time2.replace(tzinfo=None)

    # Completion times match actual execution times
    assert task1.completed_at >= completed_times[1]
    assert task2.completed_at >= completed_times[2]


@pytest.mark.asyncio
async def test_error_handling(running_scheduler, task_client):
    """Test error handling in task execution."""

    async def failing_callback():
        raise Exception("Test error")

    # Schedule failing task
    schedule_time = datetime.datetime.now()
    task_id = await running_scheduler.schedule_task(
        "Failing task",
        schedule_time,
        failing_callback,
        task_type=TaskType.ONE_TIME
    )

    # Verify initial state
    task = await task_client.get(task_id)
    assert task.task_type == SchemaTaskType.ONE_TIME
    assert not task.is_completed
    assert task.error is None

    # Wait for execution
    await asyncio.sleep(1)

    # Verify error handling
    task = await task_client.get(task_id)
    assert task.is_completed  # Task should be marked completed even if it fails
    assert task.error is not None  # Error should be recorded
    assert task.completed_at is not None
    assert task.completed_at >= schedule_time  # Completion time after schedule time
    assert "Test error" in task.error  # Error message recorded correctly


@pytest.mark.asyncio
async def test_recurring_task_continues_after_error(running_scheduler, task_client):
    """Test recurring task continues after error."""
    execution_count = 0
    error_count = 0
    execution_times = []
    error_times = []

    async def flaky_callback():
        nonlocal execution_count, error_count
        now = datetime.datetime.now()
        execution_count += 1
        execution_times.append(now)
        if execution_count % 2 == 0:
            error_count += 1
            error_times.append(now)
            raise Exception("Simulated error")

    # Schedule recurring task that fails every other time
    now = datetime.datetime.now()
    interval = datetime.timedelta(seconds=1)
    task_id = await running_scheduler.schedule_task(
        "Flaky recurring task",
        now,
        flaky_callback,
        task_type=TaskType.RECURRING,
        interval=interval,
    )

    # Verify initial task state
    task = await task_client.get(task_id)
    assert task.task_type == SchemaTaskType.RECURRING
    assert task.interval == interval
    assert not task.is_completed
    assert task.error is None
    assert task.schedule_time == now.replace(tzinfo=None)

    # Wait for multiple executions
    await asyncio.sleep(4.5)  # Should see ~4 executions
    assert execution_count >= 4  # At least 4 executions
    assert error_count >= 2  # At least 2 errors

    # Verify execution pattern
    for i in range(1, len(execution_times)):
        interval = execution_times[i] - execution_times[i - 1]
        assert 0.8 <= interval.total_seconds() <= 1.2  # Allow 200ms variance

    # Wait a bit to ensure last execution was successful
    await asyncio.sleep(1.5)

    # Verify task state after errors
    task = await task_client.get(task_id)
    assert task.task_type == SchemaTaskType.RECURRING
    assert task.interval == interval

    # Since this is a recurring task:
    # 1. It should not be marked completed as it keeps running
    # 2. Last error should be cleared after successful execution
    assert not task.is_completed
    assert task.error is None  # Error cleared after successful execution

    # Verify next schedule time is in the future
    now = datetime.datetime.now()
    assert task.schedule_time > now

    # Verify error pattern
    for error_time in error_times:
        # Each error should be on even-numbered executions
        error_idx = execution_times.index(error_time)
        assert (error_idx + 1) % 2 == 0  # error_idx is 0-based
