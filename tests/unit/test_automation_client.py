from unittest.mock import Mock, AsyncMock

import pytest

from cue.services import AutomationClient
from cue.schemas.automation import AutomationCreate, AutomationUpdate
from cue.services.transport import HTTPTransport


@pytest.fixture
def mock_schedule_components():
    return {
        "frequency": "daily",
        "start_time": "2025-12-31T00:00:00Z",
        "interval_minutes": None,
        "by_day": None,
        "by_month_day": None,
        "by_year_day": None,
        "by_month": None,
        "by_hour": None,
        "by_minute": None,
        "by_second": None,
    }


@pytest.fixture
def base_automation_data(mock_schedule_components):
    return {
        "title": "Test Automation",
        "prompt": "Test prompt",
        "is_enabled": True,
        "conversation_id": "conv_123",
        "schedule": "FREQ=DAILY;INTERVAL=1",
        "default_timezone": "UTC",
        "email_enabled": False,
        "schedule_components": mock_schedule_components,
    }


@pytest.fixture
def mock_http_transport():
    transport = Mock(spec=HTTPTransport)
    transport.request = AsyncMock()
    return transport


@pytest.fixture
def client(mock_http_transport):
    return AutomationClient(http=mock_http_transport)


def create_mock_response(base_data, **kwargs):
    """Helper function to create mock response data"""
    response_data = base_data.copy()
    for key, value in kwargs.items():
        response_data[key] = value
    return response_data


@pytest.mark.asyncio
async def test_create_automation(mock_http_transport, base_automation_data):
    mock_http_transport.request.return_value = base_automation_data
    client = AutomationClient(http=mock_http_transport)

    automation_create = AutomationCreate(**base_automation_data)
    result = await client.create(automation_create)

    # Verify the request
    mock_http_transport.request.assert_called_once_with("POST", "/automations", data=automation_create.model_dump())

    # Verify the response
    assert result.title == "Test Automation"
    assert result.prompt == "Test prompt"
    assert result.is_enabled is True
    assert result.conversation_id == "conv_123"
    assert result.schedule == "FREQ=DAILY;INTERVAL=1"


@pytest.mark.asyncio
async def test_get_automation(mock_http_transport, base_automation_data):
    mock_http_transport.request.return_value = base_automation_data
    client = AutomationClient(http=mock_http_transport)

    result = await client.get("auto_123")

    mock_http_transport.request.assert_called_once_with("GET", "/automations/auto_123")

    assert result.title == "Test Automation"
    assert result.prompt == "Test prompt"


@pytest.mark.asyncio
async def test_update_automation(mock_http_transport, base_automation_data):
    updated_data = create_mock_response(base_automation_data, title="Updated Automation", prompt="Updated prompt")
    mock_http_transport.request.return_value = updated_data
    client = AutomationClient(http=mock_http_transport)

    automation_id = "auto_123"
    automation_update = AutomationUpdate(title="Updated Automation", prompt="Updated prompt")
    result = await client.update(automation_id, automation_update)

    mock_http_transport.request.assert_called_once_with(
        "PUT", f"/automations/{automation_id}", data=automation_update.model_dump()
    )

    assert result.title == "Updated Automation"
    assert result.prompt == "Updated prompt"


@pytest.mark.asyncio
async def test_list_automations_basic(mock_http_transport, base_automation_data):
    mock_data = [base_automation_data, create_mock_response(base_automation_data, title="Second Automation")]
    mock_http_transport.request.return_value = mock_data
    client = AutomationClient(http=mock_http_transport)

    result = await client.list(skip=0, limit=10)

    mock_http_transport.request.assert_called_once_with("GET", "/automations?skip=0&limit=10")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].title == "Test Automation"
    assert result[1].title == "Second Automation"


@pytest.mark.asyncio
async def test_list_automations_with_filters(mock_http_transport, base_automation_data):
    mock_data = [base_automation_data]
    mock_http_transport.request.return_value = mock_data
    client = AutomationClient(http=mock_http_transport)

    result = await client.list(skip=0, limit=10, conversation_id="conv_123", is_enabled=True)

    mock_http_transport.request.assert_called_once_with(
        "GET", "/automations?skip=0&limit=10&conversation_id=conv_123&is_enabled=true"
    )

    assert isinstance(result, list)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_delete_automation(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = AutomationClient(http=mock_http_transport)

    result = await client.delete("auto_123")

    mock_http_transport.request.assert_called_once_with("DELETE", "/automations/auto_123")

    assert result is True


@pytest.mark.asyncio
async def test_create_automation_failed(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = AutomationClient(http=mock_http_transport)

    automation_create = AutomationCreate(
        title="Test Automation", prompt="Test prompt", conversation_id="conv_123", schedule="FREQ=DAILY;INTERVAL=1"
    )

    result = await client.create(automation_create)
    assert result is None
