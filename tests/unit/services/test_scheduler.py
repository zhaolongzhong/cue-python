"""Tests for scheduler service."""

import asyncio
import datetime

import pytest
import pytest_asyncio

from cue.services.scheduler import SchedulerService


@pytest.fixture
def scheduler():
    """Provide scheduler service instance."""
    service = SchedulerService()
    return service


@pytest_asyncio.fixture
async def running_scheduler(scheduler):
    """Provide running scheduler service."""
    await scheduler.start()
    try:
        yield scheduler
    finally:
        await scheduler.stop()


async def mock_callback(*args, **kwargs):
    """Mock callback for testing."""
    pass


def test_scheduler_singleton():
    """Test scheduler is singleton."""
    s1 = SchedulerService()
    s2 = SchedulerService()
    assert s1 is s2


@pytest.mark.asyncio
async def test_schedule_task(scheduler):
    """Test scheduling a task."""
    schedule_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
    task_id = scheduler.schedule_task("Test task", schedule_time, mock_callback)

    task = scheduler.get_task(task_id)
    assert task is not None
    assert task.instruction == "Test task"
    assert task.schedule_time == schedule_time
    assert not task.is_completed


@pytest.mark.asyncio
async def test_cancel_task(scheduler):
    """Test cancelling a task."""
    schedule_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
    task_id = scheduler.schedule_task("Test task", schedule_time, mock_callback)

    assert scheduler.cancel_task(task_id)
    assert scheduler.get_task(task_id) is None


@pytest.mark.asyncio
async def test_task_execution(running_scheduler):
    """Test task execution."""
    executed = False

    async def test_callback():
        nonlocal executed
        executed = True

    # Schedule task for immediate execution
    schedule_time = datetime.datetime.now()
    running_scheduler.schedule_task("Test execution", schedule_time, test_callback)

    # Wait for execution
    await asyncio.sleep(2)
    assert executed


@pytest.mark.asyncio
async def test_multiple_tasks(running_scheduler):
    """Test handling multiple tasks."""
    results = []

    async def callback(value):
        results.append(value)

    now = datetime.datetime.now()

    # Schedule multiple tasks
    running_scheduler.schedule_task("Task 1", now + datetime.timedelta(milliseconds=100), callback, 1)
    running_scheduler.schedule_task("Task 2", now + datetime.timedelta(milliseconds=200), callback, 2)

    # Wait for execution
    await asyncio.sleep(1)
    assert results == [1, 2]


@pytest.mark.asyncio
async def test_error_handling(running_scheduler):
    """Test error handling in task execution."""

    async def failing_callback():
        raise Exception("Test error")

    # Schedule failing task
    schedule_time = datetime.datetime.now()
    task_id = running_scheduler.schedule_task("Failing task", schedule_time, failing_callback)

    # Wait for execution
    await asyncio.sleep(1)
    task = running_scheduler.get_task(task_id)
    assert task.is_completed  # Task should be marked completed even if it fails
