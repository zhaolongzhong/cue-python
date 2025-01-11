import pytest

from src.app.api.deps import validate_token
from src.app.core.config import get_settings
from tests.client_wrapper import ClientWrapper


@pytest.mark.anyio
async def test_user_access_token():
    client = ClientWrapper(
        client_id="test_client",
        user_id="default_user_id",
        auto_connect=False,
    )

    client.login()
    assert client.access_token, "access token should not be none"
    user_id = await validate_token(client.access_token, get_settings())
    assert user_id == "default_user_id"
