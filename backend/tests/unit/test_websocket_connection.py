from tests.client_wrapper import ClientWrapper, is_client_connect, is_client_disconnect, wait_for_message_with_predicate
from src.app.schemas.event_message import (
    EventMessageType,
)


def test_relationship_persistence_with_assistant():
    """Test that relationships persist correctly when user disconnects and reconnects with active assistant."""

    bob = ClientWrapper(
        client_id="client_bob",
        user_id="user_alice",
        assistant_id="asst_bob",
        auto_connect=True,
    )

    alice = ClientWrapper(
        client_id="client_alice",
        user_id="user_alice",
        auto_connect=True,
    )

    try:
        # Assistant should receive user's connect notification
        response = bob.receive_message()
        assert response["type"] == EventMessageType.CLIENT_CONNECT
        assert response["client_id"] == alice.client_id

        # Verify initial messaging works
        bob.send_message(recipient=alice.user_id, message="Initial message")
        response = alice.receive_message()
        assert response["payload"]["message"] == "Initial message"

        # Disconnect user
        alice.disconnect()

        # Assistant should receive disconnect notification
        response = bob.receive_message()
        assert response["type"] == EventMessageType.CLIENT_DISCONNECT
        assert response["client_id"] == alice.client_id

        # Reconnect user
        alice.connect()

        # Assistant should receive reconnect notification
        response = bob.receive_message()
        assert response["type"] == EventMessageType.CLIENT_CONNECT
        assert response["client_id"] == alice.client_id

        # Verify messaging still works after reconnection
        bob.send_message(recipient=alice.user_id, message="Message after reconnect")
        response = alice.receive_message()
        assert response["payload"]["message"] == "Message after reconnect"

        # Verify user can still message assistant
        alice.send_message(recipient=bob.assistant_id, message="User message after reconnect")
        response = bob.receive_message()
        assert response["payload"]["message"] == "User message after reconnect"

    finally:
        alice.clean_up()
        bob.clean_up()


def test_relationship_persistence_multiple_reconnects():
    """Test relationships persist through multiple user reconnections."""
    alice = ClientWrapper(
        client_id="client_alice",
        user_id="user_alice",
        auto_connect=False,
    )

    bob = ClientWrapper(
        client_id="client_bob",
        user_id="user_alice",
        assistant_id="asst_bob",
        auto_connect=False,
    )

    try:
        alice.connect()
        bob.connect()
        # Initial connection check
        response = alice.receive_message()
        assert response["type"] == EventMessageType.CLIENT_CONNECT

        # Test multiple disconnect/reconnect cycles
        for i in range(3):  # Test 3 cycles
            # Verify messaging works
            test_message = f"Test message {i}"
            bob.send_message(recipient=alice.user_id, message=test_message)
            response = alice.receive_message()
            assert response["payload"]["message"] == test_message

            # Disconnect user
            alice.disconnect()
            response = bob.receive_message()
            assert response["type"] == EventMessageType.CLIENT_DISCONNECT

            # Reconnect user with new client
            alice = ClientWrapper(
                client_id=f"user_client_id_{i}",
                user_id=alice.user_id,
                auto_connect=True,
            )

            # Assistant should receive reconnect notification
            response = bob.receive_message()
            assert response["type"] == EventMessageType.CLIENT_CONNECT

            # Verify bidirectional messaging after each reconnect
            bob.send_message(recipient=alice.user_id, message=f"Assistant message after reconnect {i}")
            response = alice.receive_message()
            assert response["payload"]["message"] == f"Assistant message after reconnect {i}"

            alice.send_message(recipient=bob.assistant_id, message=f"User message after reconnect {i}")
            response = bob.receive_message()
            assert response["payload"]["message"] == f"User message after reconnect {i}"

    finally:
        alice.clean_up()
        bob.clean_up()


def test_relationship_persistence_with_multiple_assistants():
    """Test relationships persist correctly with multiple assistants when user reconnects."""
    # Connect two assistants
    bob = ClientWrapper(
        client_id="client_bob",
        user_id="user_alice",
        assistant_id="asst_bob",
        auto_connect=True,
    )

    charlie = ClientWrapper(
        client_id="client_charlie",
        user_id="user_alice",
        assistant_id="asst_charlie",
        auto_connect=True,
    )

    alice = ClientWrapper(
        client_id="client_alice",
        user_id="user_alice",
        auto_connect=True,
    )

    try:
        # Both assistants should receive user's connect notification
        response1 = bob.receive_message()
        assert response1["type"] == EventMessageType.CLIENT_CONNECT
        response2 = charlie.receive_message()
        assert response2["type"] == EventMessageType.CLIENT_CONNECT

        # Verify initial messaging works with both assistants
        bob.send_message(recipient=alice.user_id, message="Message from assistant 1")
        response = alice.receive_message()
        assert response["payload"]["message"] == "Message from assistant 1"

        charlie.send_message(recipient=alice.user_id, message="Message from assistant 2")
        response = alice.receive_message()
        assert response["payload"]["message"] == "Message from assistant 2"

        # Disconnect user
        alice.disconnect()

        disconnect_msg_bob = wait_for_message_with_predicate(bob, is_client_disconnect("client_alice"), timeout=5.0)
        assert disconnect_msg_bob is not None, "Bob didn't receive disconnect message"

        disconnect_msg_charlie = wait_for_message_with_predicate(
            charlie, is_client_disconnect("client_alice"), timeout=5.0
        )
        assert disconnect_msg_charlie is not None, "Charlie didn't receive disconnect message"

        # Reconnect alice with new client
        alice = ClientWrapper(
            client_id="client_alice",
            user_id="user_alice",
            auto_connect=True,
        )

        # Verify connect messages are received
        connect_msg_bob = wait_for_message_with_predicate(bob, is_client_connect("client_alice"), timeout=5.0)
        assert connect_msg_bob is not None, "Bob didn't receive connect message"
        connect_msg_charlie = wait_for_message_with_predicate(charlie, is_client_connect("client_alice"), timeout=5.0)
        assert connect_msg_charlie is not None, "Charlie didn't receive disconnect message"

    finally:
        # Cleanup
        bob.clean_up()
        charlie.clean_up()
        alice.clean_up()
