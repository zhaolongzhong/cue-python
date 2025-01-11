from tests.client_wrapper import ClientWrapper, wait_for_message_with_predicate
from src.app.schemas.event_message import (
    EventMessage,
    MessagePayload,
    EventMessageType,
)


def test_ping_pong():
    client = ClientWrapper(
        client_id="test_client",
        user_id="default_user_id",
        auto_connect=True,
    )

    try:
        client.send_ping()
        response = client.receive_message()
        assert response["type"] == EventMessageType.PONG
        assert response["client_id"] == "test_client"
        assert response["payload"]["type"] == EventMessageType.PONG
    finally:
        client.disconnect()


def test_two_users_chat():
    alice = ClientWrapper(client_id="alice_client_id", user_id="alice_user_id", auto_connect=True)
    bob = ClientWrapper(
        client_id="bob_client_id",
        user_id="alice_user_id",
        assistant_id="bob_user_id",
        auto_connect=True,
    )

    # Alice receives Bob's connect notification
    response = alice.receive_message()
    assert response["type"] == EventMessageType.CLIENT_CONNECT
    assert response["client_id"] == bob.client_id

    # Test message from Alice to Bob
    alice.send_message(bob.assistant_id, "hello bob, this is alice")
    response = bob.receive_message()
    assert response["type"] == EventMessageType.GENERIC
    assert response["client_id"] == "alice_client_id"
    assert response["payload"]["message"] == "hello bob, this is alice"

    # Test message from Bob to Alice
    bob.send_message(alice.user_id, "hey alice")
    response = alice.receive_message()
    assert response["type"] == EventMessageType.GENERIC
    assert response["client_id"] == bob.client_id
    assert response["payload"]["message"] == "hey alice"

    alice.disconnect()
    assert not alice.is_connected

    # Bob should receive Alice's disconnect notification
    response = bob.receive_message()
    assert response["type"] == EventMessageType.CLIENT_DISCONNECT
    assert response["client_id"] == alice.client_id

    # Alice reconnect when Bob is already online
    alice.connect()
    assert alice.is_connected

    # Bob should receive Alice's connect notification
    response = bob.receive_message()
    assert response["type"] == EventMessageType.CLIENT_CONNECT
    assert response["client_id"] == alice.client_id

    # Send a message from Bob to Alice
    bob.send_message(alice.user_id, "hello alice, this is bob")

    # Alice should receive the message from Bob since Bob still keeps the relationship
    response = alice.receive_message()
    assert response["type"] == EventMessageType.GENERIC
    assert response["client_id"] == bob.client_id
    assert response["payload"]["message"] == "hello alice, this is bob"

    # Send a message from Alice to Bob
    alice.send_message(bob.assistant_id, "hello bob, this is alice again")

    # Bob should receive the message
    response = bob.receive_message()
    assert response["type"] == EventMessageType.GENERIC
    assert response["client_id"] == alice.client_id
    assert response["payload"]["message"] == "hello bob, this is alice again"

    # Clean up
    alice.clean_up()
    bob.clean_up()
    assert not alice.is_connected
    assert not bob.is_connected


def test_three_users_chat():
    alice = ClientWrapper(
        client_id="client_alice",
        user_id="user_alice",
        auto_connect=True,
    )
    bob = ClientWrapper(
        client_id="client_bob",
        user_id=alice.user_id,
        assistant_id="asst_bob",
        auto_connect=False,
    )
    charlie = ClientWrapper(
        client_id="client_charlie",
        user_id=alice.user_id,
        assistant_id="asst_charlie",
        auto_connect=False,
    )

    # Alice receives Bob's connect notification

    bob.connect()
    response = alice.receive_message()
    assert response["type"] == EventMessageType.CLIENT_CONNECT
    assert response["client_id"] == bob.client_id

    # Alice receives Charles' connect notification
    charlie.connect()
    response = alice.receive_message()
    assert response["type"] == EventMessageType.CLIENT_CONNECT
    assert response["client_id"] == charlie.client_id

    # Send a message from Alice to Bob
    alice.send_message(bob.assistant_id, "hello bob, this is alice")

    # Bob should receive the message
    response = bob.receive_message()
    response = wait_for_message_with_predicate(
        bob, lambda msg: (msg["type"] == EventMessageType.GENERIC and msg["client_id"] == "client_alice"), timeout=5.0
    )
    assert response is not None, "Bob didn't receive the message"
    assert response["payload"]["message"] == "hello bob, this is alice"

    # Charles should not receive any message
    assert not charlie.has_message(), "Charles should not receive messages intended for Bob"

    # Test that we can still send messages to Charles
    alice.send_message(charlie.assistant_id, "hello charlie, this is alice")
    response = wait_for_message_with_predicate(
        charlie,
        lambda msg: (msg["type"] == EventMessageType.GENERIC and msg["client_id"] == "client_alice"),
        timeout=5.0,
    )
    assert response["type"] == EventMessageType.GENERIC
    assert response["payload"]["message"] == "hello charlie, this is alice"

    # Send a message from Bob to Alice
    bob.send_message(alice.user_id, "hello alice, this is bob")

    # Alice should receive the message from Bob
    response = wait_for_message_with_predicate(
        alice, lambda msg: (msg["type"] == EventMessageType.GENERIC and msg["client_id"] == "client_bob"), timeout=5.0
    )
    assert response["payload"]["message"] == "hello alice, this is bob"

    # Send a message from Charlie to Alice
    charlie.send_message(alice.user_id, "hello alice, this is charlie")

    # Alice should receive the message from Charlie
    response = wait_for_message_with_predicate(
        alice,
        lambda msg: (msg["type"] == EventMessageType.GENERIC and msg["client_id"] == "client_charlie"),
        timeout=5.0,
    )
    assert response["payload"]["message"] == "hello alice, this is charlie"

    event = EventMessage(
        client_id=bob.client_id,
        type=EventMessageType.ASSISTANT.value,
        payload=MessagePayload(
            type=EventMessageType.ASSISTANT, recipient=charlie.assistant_id, message="hello charlie, this is bob"
        ),
    )
    bob.send_event(event)
    response = wait_for_message_with_predicate(
        charlie,
        lambda msg: (msg["type"] == EventMessageType.ASSISTANT and msg["client_id"] == "client_bob"),
        timeout=5.0,
    )
    assert response["payload"]["message"] == "hello charlie, this is bob"

    # Clean up
    alice.clean_up()
    bob.clean_up()
    charlie.clean_up()
    assert not alice.is_connected
    assert not bob.is_connected
    assert not charlie.is_connected
