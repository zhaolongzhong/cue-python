"""Tests for scheduler service with task client."""
import asyncio
import datetime

import pytest
import pytest_asyncio

from cue.services.scheduler import SchedulerService
from cue.services.task_client import TaskClient
from tests.mocks.mock_http_transport import MockHTTPTransport


# Test callback for tasks
async def mock_callback(*args, **kwargs):
    """Mock callback for testing."""
    pass


# Test fixtures
@pytest.fixture
def mock_transport():
    """Provide mock HTTP transport."""
    return MockHTTPTransport()


@pytest.fixture
def task_client(mock_transport):
    """Provide task client with mock transport."""
    return TaskClient(mock_transport)


@pytest.fixture
def scheduler(task_client):
    """Provide scheduler service with mock task client."""
    service = SchedulerService(task_client)
    return service


@pytest_asyncio.fixture
async def running_scheduler(scheduler):
    """Provide running scheduler service."""
    await scheduler.start()
    try:
        yield scheduler
    finally:
        await scheduler.stop()


def test_scheduler_singleton():
    """Test scheduler is singleton."""
    transport = MockHTTPTransport()
    client = TaskClient(transport)
    s1 = SchedulerService(client)
    s2 = SchedulerService(client)
    assert s1 is s2


@pytest.mark.asyncio
async def test_schedule_task(scheduler):
    """Test scheduling a task."""
    schedule_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
    task_id = await scheduler.schedule_task(
        "Test task",
        schedule_time,
        mock_callback,
        "arg1",
        kwarg1="value1"
    )

    assert task_id is not None
    task = await scheduler.task_client.get(task_id)
    assert task is not None
    assert task.instruction == "Test task"
    assert task.schedule_time == schedule_time
    assert not task.is_completed
    assert task.metadata.callback_args == ["arg1"]
    assert task.metadata.callback_kwargs == {"kwarg1": "value1"}


@pytest.mark.asyncio
async def test_task_execution(running_scheduler):
    """Test task execution and completion."""
    executed = False

    async def test_callback():
        nonlocal executed
        executed = True

    # Schedule task for immediate execution
    now = datetime.datetime.now()
    task_id = await running_scheduler.schedule_task(
        "Test execution",
        now,
        test_callback
    )

    # Wait for execution
    await asyncio.sleep(2)

    # Verify task completion
    task = await running_scheduler.task_client.get(task_id)
    assert executed
    assert task.is_completed
    assert task.completed_at is not None


@pytest.mark.asyncio
async def test_recurring_task(running_scheduler):
    """Test recurring task execution."""
    executions = []

    async def test_callback():
        executions.append(datetime.datetime.now())

    # Schedule recurring task every second
    now = datetime.datetime.now()
    await running_scheduler.schedule_task(
        "Recurring test",
        now,
        test_callback,
        interval=datetime.timedelta(seconds=1)
    )

    # Wait for multiple executions
    await asyncio.sleep(3.5)  # Should see ~3 executions
    assert len(executions) >= 3

    # Verify intervals between executions
    for i in range(1, len(executions)):
        interval = executions[i] - executions[i-1]
        assert 0.8 <= interval.total_seconds() <= 1.2  # Allow 200ms variance


@pytest.mark.asyncio
async def test_error_handling(running_scheduler):
    """Test error handling in task execution."""
    async def failing_callback():
        raise ValueError("Test error")

    # Schedule failing task
    now = datetime.datetime.now()
    task_id = await running_scheduler.schedule_task(
        "Failing task",
        now,
        failing_callback
    )

    # Wait for execution
    await asyncio.sleep(1)

    # Verify error is recorded
    task = await running_scheduler.task_client.get(task_id)
    assert task.is_completed
    assert task.error is not None
    assert "Test error" in task.error


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
            raise ValueError("Simulated error")

    # Schedule recurring task that fails every other time
    now = datetime.datetime.now()
    task_id = await running_scheduler.schedule_task(
        "Flaky recurring task",
        now,
        flaky_callback,
        interval=datetime.timedelta(seconds=1)
    )

    # Wait for multiple executions
    await asyncio.sleep(4.5)  # Should see ~4 executions

    # Verify task kept running despite errors
    assert execution_count >= 4  # At least 4 executions
    assert error_count >= 2  # At least 2 errors

    # Verify task is still scheduled
    task = await running_scheduler.task_client.get(task_id)
    assert not task.is_completed  # Recurring tasks are never "completed"
    assert task.schedule_time > datetime.datetime.now()  # Next execution is in the future
