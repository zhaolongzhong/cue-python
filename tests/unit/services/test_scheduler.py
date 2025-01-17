"""Tests for scheduler service."""

import asyncio
import datetime

import pytest
import pytest_asyncio

from cue.services.scheduler import TaskType, SchedulerService


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
async def test_one_time_task(scheduler):
    """Test scheduling a one-time task."""
    schedule_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
    task_id = scheduler.schedule_task("Test task", schedule_time, mock_callback, task_type=TaskType.ONE_TIME)

    task = scheduler.get_task(task_id)
    assert task is not None
    assert task.instruction == "Test task"
    assert task.schedule_time == schedule_time
    assert task.task_type == TaskType.ONE_TIME
    assert not task.is_completed


@pytest.mark.asyncio
async def test_recurring_task(running_scheduler):
    """Test recurring task execution."""
    results = []

    async def test_callback():
        results.append(datetime.datetime.now())

    # Schedule recurring task every second
    now = datetime.datetime.now()
    running_scheduler.schedule_task(
        "Recurring test", now, test_callback, task_type=TaskType.RECURRING, interval=datetime.timedelta(seconds=1)
    )

    # Wait for multiple executions
    await asyncio.sleep(3.5)  # Should see ~3 executions
    assert len(results) >= 3

    # Verify intervals between executions
    for i in range(1, len(results)):
        interval = results[i] - results[i - 1]
        assert 0.8 <= interval.total_seconds() <= 1.2  # Allow 200ms variance


@pytest.mark.asyncio
async def test_recurring_task_validation(scheduler):
    """Test recurring task requires interval."""
    schedule_time = datetime.datetime.now()

    with pytest.raises(ValueError, match="Interval is required for recurring tasks"):
        scheduler.schedule_task(
            "Invalid recurring task",
            schedule_time,
            mock_callback,
            task_type=TaskType.RECURRING,  # Missing interval
        )


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


@pytest.mark.asyncio
async def test_recurring_task_continues_after_error(running_scheduler):
    """Test recurring task continues after error."""
    execution_count = 0
    error_count = 0

    async def flaky_callback():
        nonlocal execution_count, error_count
        execution_count += 1
        if execution_count % 2 == 0:
            error_count += 1
            raise Exception("Simulated error")

    # Schedule recurring task that fails every other time
    now = datetime.datetime.now()
    running_scheduler.schedule_task(
        "Flaky recurring task",
        now,
        flaky_callback,
        task_type=TaskType.RECURRING,
        interval=datetime.timedelta(seconds=1),
    )

    # Wait for multiple executions
    await asyncio.sleep(4.5)  # Should see ~4 executions
    assert execution_count >= 4  # At least 4 executions
    assert error_count >= 2  # At least 2 errors
