"""Bridge-free deterministic tests for the Streamer V1 source boundary."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Optional

import pytest

from pvzrl_streamer_source import (
    BoundedStreamMessageBuffer,
    DeterministicStreamCommandSource,
    ScriptedStreamSourceRecord,
    StreamSourceMessage,
)
from pvzrl_twitch import (
    EventSubSessionWatchdog,
    SubscriptionResult,
    TokenValidation,
    TwitchConfigurationError,
    TwitchCredentials,
    TwitchEventSubProtocol,
    TwitchEventSubSource,
    TwitchHttpError,
    UrllibTwitchEventSubHttpClient,
)


BROADCASTER_ID = "broadcaster-42"
READER_ID = "reader-24"
TOKEN = "test-token-must-never-escape"
HASH_SECRET = b"test-only-viewer-hash-secret-32b"


def _credentials() -> TwitchCredentials:
    return TwitchCredentials(
        client_id="client-7",
        access_token=TOKEN,
        broadcaster_user_id=BROADCASTER_ID,
        eventsub_user_id=READER_ID,
        viewer_hash_secret=HASH_SECRET,
    )


def _metadata(message_type: str, message_id: str) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "message_type": message_type,
        "message_timestamp": "2026-08-08T12:00:00.000000Z",
    }


def _welcome(session_id: str, *, timeout_seconds: float = 30.0) -> str:
    return json.dumps(
        {
            "metadata": _metadata("session_welcome", f"welcome-{session_id}"),
            "payload": {
                "session": {
                    "id": session_id,
                    "status": "connected",
                    "keepalive_timeout_seconds": timeout_seconds,
                }
            },
        }
    )


def _keepalive(message_id: str = "keepalive-1") -> str:
    return json.dumps(
        {
            "metadata": _metadata("session_keepalive", message_id),
            "payload": {},
        }
    )


def _notification(
    delivery_id: str,
    event_id: str,
    *,
    chatter_id: str = "raw-viewer-id-99",
    chatter_name: str = "RawViewerName",
    text: str = "!plant sunflower 2 4",
) -> str:
    metadata = _metadata("notification", delivery_id)
    metadata.update(
        {
            "subscription_type": "channel.chat.message",
            "subscription_version": "1",
        }
    )
    return json.dumps(
        {
            "metadata": metadata,
            "payload": {
                "subscription": {
                    "id": "subscription-1",
                    "status": "enabled",
                    "type": "channel.chat.message",
                    "version": "1",
                    "condition": {
                        "broadcaster_user_id": BROADCASTER_ID,
                        "user_id": READER_ID,
                    },
                },
                "event": {
                    "broadcaster_user_id": BROADCASTER_ID,
                    "chatter_user_id": chatter_id,
                    "chatter_user_name": chatter_name,
                    "message_id": event_id,
                    "message": {"text": text},
                },
            },
        }
    )


def _reconnect(delivery_id: str, reconnect_url: str) -> str:
    return json.dumps(
        {
            "metadata": _metadata("session_reconnect", delivery_id),
            "payload": {"session": {"reconnect_url": reconnect_url}},
        }
    )


def _revocation(delivery_id: str, status: str) -> str:
    return json.dumps(
        {
            "metadata": _metadata("revocation", delivery_id),
            "payload": {"subscription": {"status": status}},
        }
    )


class _FakeHttpClient:
    def __init__(self, *validation_results: object) -> None:
        default = TokenValidation(
            client_id="client-7",
            user_id=READER_ID,
            scopes=("user:read:chat",),
            expires_in=14_400,
        )
        self._validation_results: Deque[object] = deque(validation_results or (default,))
        self._last_validation: object = self._validation_results[-1]
        self.validate_calls = 0
        self.subscription_sessions: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def validate_token(self, credentials: TwitchCredentials) -> TokenValidation:
        assert credentials.access_token == TOKEN
        with self._lock:
            self.validate_calls += 1
            result = self._validation_results.popleft() if self._validation_results else self._last_validation
            self._last_validation = result
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, TokenValidation)
        return result

    def create_chat_subscription(
        self,
        credentials: TwitchCredentials,
        *,
        session_id: str,
        eventsub_user_id: str,
    ) -> SubscriptionResult:
        assert credentials.access_token == TOKEN
        with self._lock:
            self.subscription_sessions.append((session_id, eventsub_user_id))
            index = len(self.subscription_sessions)
        return SubscriptionResult(
            subscription_id=f"subscription-{index}",
            status="enabled",
            session_id=session_id,
        )


class _FakeWebSocket:
    def __init__(self, *frames: object) -> None:
        self._frames: Deque[object] = deque(frames)
        self._lock = threading.Lock()
        self.closed = threading.Event()
        self.blocked_in_recv = threading.Event()
        self.welcome_received = threading.Event()

    async def recv(self) -> Any:
        with self._lock:
            item: Optional[object] = self._frames.popleft() if self._frames else None
        if item is None:
            self.blocked_in_recv.set()
            await _block_forever()
            raise AssertionError("unreachable")
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str) and '"session_welcome"' in item:
            self.welcome_received.set()
        return item

    async def close(self) -> None:
        self.closed.set()


async def _block_forever() -> None:
    import asyncio

    await asyncio.Future()


class _FakeConnector:
    def __init__(self, *connections: _FakeWebSocket) -> None:
        self._connections: Deque[_FakeWebSocket] = deque(connections)
        self.urls: list[str] = []
        self._lock = threading.Lock()

    async def __call__(self, url: str) -> _FakeWebSocket:
        with self._lock:
            self.urls.append(url)
            if not self._connections:
                raise ConnectionError("no_fake_connection")
            return self._connections.popleft()


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "condition did not become true before timeout"


def _source(http: _FakeHttpClient, connector: _FakeConnector, **kwargs: Any) -> TwitchEventSubSource:
    return TwitchEventSubSource(
        _credentials(),
        http_client=http,
        websocket_connector=connector,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        random_value=lambda: 0.0,
        **kwargs,
    )


def test_protocol_hashes_identity_deduplicates_and_bounds_untrusted_text() -> None:
    credentials = _credentials()
    protocol = TwitchEventSubProtocol(credentials, max_command_chars=32, max_command_bytes=64)
    result = protocol.parse(
        _notification("delivery-1", "event-1", text="!slot 3 2 4"),
        received_monotonic=12.5,
    )
    assert result.kind == "notification"
    assert result.message is not None
    expected_hash = hmac.new(HASH_SECRET, b"raw-viewer-id-99", hashlib.sha256).hexdigest()
    assert result.message.viewer_hash == expected_hash
    assert result.message.received_monotonic == 12.5
    assert "!slot" not in repr(result.message)
    serialized = repr(result.message) + json.dumps(result.message.__dict__)
    assert "raw-viewer-id-99" not in serialized
    assert "RawViewerName" not in serialized
    assert TOKEN not in repr(credentials)
    assert HASH_SECRET.decode("utf-8") not in repr(credentials)
    assert BROADCASTER_ID not in repr(credentials)
    assert READER_ID not in repr(credentials)
    with pytest.raises(TwitchConfigurationError) as invalid_token:
        TwitchCredentials(
            client_id="client-7",
            access_token="raw-secret\nheader-injection",
            broadcaster_user_id=BROADCASTER_ID,
            viewer_hash_secret=HASH_SECRET,
        )
    assert "raw-secret" not in str(invalid_token.value)

    duplicate_delivery = protocol.parse(
        _notification("delivery-1", "event-2"),
        received_monotonic=13.0,
    )
    duplicate_event = protocol.parse(
        _notification("delivery-2", "event-1"),
        received_monotonic=13.5,
    )
    assert duplicate_delivery.kind == "duplicate"
    assert duplicate_event.kind == "duplicate"

    oversized = protocol.parse(
        _notification("delivery-3", "event-3", text="x" * 33),
        received_monotonic=14.0,
    )
    malformed_unicode = protocol.parse(
        _notification("delivery-4", "event-4", text="\ud800"),
        received_monotonic=14.5,
    )
    assert oversized.kind == "malformed"
    assert malformed_unicode.kind == "malformed"
    ordinary_chat = protocol.parse(
        _notification("delivery-5", "event-5", text="hello streamer"),
        received_monotonic=15.0,
    )
    assert ordinary_chat.kind == "ignored"
    assert ordinary_chat.error_code == "non_command_chat"
    assert ordinary_chat.message is None


def test_http_authorization_schemes_match_twitch_endpoints() -> None:
    class CapturingHttpClient(UrllibTwitchEventSubHttpClient):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[Any] = []

        def _request_json(self, request: Any, *, expected_status: int) -> dict[str, Any]:
            self.requests.append(request)
            if "oauth2/validate" in request.full_url:
                return {
                    "client_id": "client-7",
                    "user_id": READER_ID,
                    "scopes": ["user:read:chat"],
                    "expires_in": 14_400,
                }
            assert expected_status == 202
            return {
                "data": [
                    {
                        "id": "subscription-http",
                        "status": "enabled",
                        "transport": {"session_id": "session-http"},
                    }
                ]
            }

    http = CapturingHttpClient()
    http.validate_token(_credentials())
    http.create_chat_subscription(
        _credentials(),
        session_id="session-http",
        eventsub_user_id=READER_ID,
    )
    assert http.requests[0].get_header("Authorization") == f"OAuth {TOKEN}"
    assert http.requests[1].get_header("Authorization") == f"Bearer {TOKEN}"


def test_protocol_rejects_untrusted_reconnect_hosts() -> None:
    protocol = TwitchEventSubProtocol(_credentials())
    malicious = protocol.parse(
        _reconnect("reconnect-bad", "wss://example.invalid/ws?session=raw-secret"),
        received_monotonic=1.0,
    )
    twitch = protocol.parse(
        _reconnect(
            "reconnect-good",
            "wss://eventsub.wss.twitch.tv/ws?reconnect=opaque-value",
        ),
        received_monotonic=2.0,
    )
    assert malicious.kind == "malformed"
    assert twitch.kind == "reconnect"


def test_bounded_phase_gate_and_privacy_safe_deterministic_source() -> None:
    old_hash = "a" * 64
    records = (
        ScriptedStreamSourceRecord("!wait", viewer_hash=old_hash),
        ScriptedStreamSourceRecord("!plant sunflower 2 4", local_viewer_id="local-alice"),
    )
    source = DeterministicStreamCommandSource(
        records,
        viewer_hash_secret=HASH_SECRET,
        queue_capacity=1,
        monotonic=lambda: 9.0,
    )
    assert "local-alice" not in repr(records[1])
    assert "local-alice" not in repr(source)
    source.start()
    diagnostics = source.get_diagnostics()
    assert diagnostics["stream_source_queue_depth"] == 1
    assert diagnostics["stream_source_messages_discarded_queue_full"] == 0
    assert diagnostics["stream_source_script_records_pending"] == 1
    assert "local-alice" not in json.dumps(diagnostics)
    first = source.drain_messages()
    assert [message.viewer_hash for message in first] == [old_hash]
    scripted_local_message = source.drain_messages()[0]
    assert scripted_local_message.viewer_hash == hmac.new(
        HASH_SECRET,
        b"local-alice",
        hashlib.sha256,
    ).hexdigest()
    assert source.submit("!wait", local_viewer_id="local-alice")
    local_message = source.drain_messages()[0]
    assert local_message.viewer_hash == hmac.new(
        HASH_SECRET,
        b"local-alice",
        hashlib.sha256,
    ).hexdigest()
    assert "local-alice" not in repr(local_message)

    source.set_accepting(False, reason="autonomous_evaluation")
    assert not source.submit(
        "!slot 1 1 1",
        local_viewer_id="eval-viewer",
        delivery_id="eval-delivery",
        event_id="eval-event",
    )
    source.set_accepting(True, reason="stream_train")
    assert source.drain_messages() == []
    assert not source.submit(
        "!slot 1 1 1",
        local_viewer_id="eval-viewer",
        delivery_id="eval-delivery",
        event_id="eval-event",
    )
    assert source.submit("!wait", viewer_hash="b" * 64)
    message = source.drain_messages()[0]
    assert message.platform == "mock"
    assert message.viewer_hash == "b" * 64
    assert "eval-viewer" not in repr(message)
    assert source.stop()
    assert not source.submit("!wait", viewer_hash="c" * 64)

    buffer = BoundedStreamMessageBuffer(1)
    _, stale_epoch = buffer.gate_snapshot()
    buffer.set_accepting(False, reason="evaluation")
    stale = StreamSourceMessage("mock", "d", "e", "f" * 64, "!wait", 1.0)
    assert not buffer.publish(stale, gate_epoch=stale_epoch)
    assert buffer.diagnostics()["stream_source_messages_discarded_phase_stale"] == 1


def test_welcome_subscription_notifications_malformed_and_shutdown() -> None:
    connection = _FakeWebSocket(
        _welcome("session-1"),
        _keepalive(),
        "{not-json",
        _notification("delivery-10", "event-10"),
    )
    http = _FakeHttpClient()
    connector = _FakeConnector(connection)
    source = _source(http, connector)
    source.start()
    _wait_until(lambda: source.get_diagnostics()["twitch_notifications_enqueued"] == 1)
    diagnostics = source.get_diagnostics()
    diagnostics_text = json.dumps(diagnostics)
    assert http.subscription_sessions == [("session-1", READER_ID)]
    assert diagnostics["twitch_keepalives_received"] == 1
    assert diagnostics["twitch_malformed_events"] == 1
    assert "raw-viewer-id-99" not in diagnostics_text
    assert "RawViewerName" not in diagnostics_text
    assert "!plant" not in diagnostics_text
    assert len(source.drain_messages()) == 1
    assert source.stop(timeout_seconds=1.0)
    assert connection.closed.is_set()


def test_twitch_eval_phase_discards_chat_without_a_resume_backlog() -> None:
    connection = _FakeWebSocket(
        _welcome("session-eval"),
        _notification("delivery-eval", "event-eval"),
    )
    source = _source(_FakeHttpClient(), _FakeConnector(connection), accepting=False)
    source.start()
    _wait_until(lambda: source.get_diagnostics()["twitch_notifications_received"] == 1)
    diagnostics = source.get_diagnostics()
    assert diagnostics["twitch_phase_accepting"] is False
    assert diagnostics["twitch_network_available"] is True
    assert diagnostics["stream_source_messages_discarded_not_accepting"] == 1
    assert source.drain_messages() == []
    source.set_accepting(True, reason="resume_stream_train")
    assert source.drain_messages() == []
    assert source.stop(timeout_seconds=1.0)


def test_twitch_requested_reconnect_handoff_uses_exact_url_without_resubscribe() -> None:
    reconnect_url = "wss://eventsub.wss.twitch.tv/ws?reconnect=opaque-value-123"
    old_connection = _FakeWebSocket(
        _welcome("session-old"),
        _reconnect("reconnect-1", reconnect_url),
    )
    new_connection = _FakeWebSocket(
        _welcome("session-new"),
        _notification("delivery-new", "event-new"),
    )
    http = _FakeHttpClient()
    connector = _FakeConnector(old_connection, new_connection)
    source = _source(http, connector)
    source.start()
    _wait_until(lambda: source.get_diagnostics()["twitch_notifications_enqueued"] == 1)
    assert connector.urls[1] == reconnect_url
    assert http.subscription_sessions == [("session-old", READER_ID)]
    assert new_connection.welcome_received.is_set()
    assert old_connection.closed.is_set()
    assert source.get_diagnostics()["twitch_reconnect_handoffs"] == 1
    assert "opaque-value-123" not in json.dumps(source.get_diagnostics())
    assert source.stop(timeout_seconds=1.0)


def test_cancelled_reconnect_handoff_closes_registered_new_socket_before_shutdown() -> None:
    import asyncio

    reconnect_url = "wss://eventsub.wss.twitch.tv/ws?reconnect=cancelled-handoff"
    blocked_handoff = _FakeWebSocket()

    class _FailAfterHandoffStarts(_FakeWebSocket):
        def __init__(self, *frames: object) -> None:
            super().__init__(*frames)
            self.recv_count = 0

        async def recv(self) -> Any:
            self.recv_count += 1
            if self.recv_count <= 2:
                return await super().recv()
            while not blocked_handoff.blocked_in_recv.is_set():
                await asyncio.sleep(0.001)
            raise ConnectionError("old_socket_failed_during_handoff")

    old_connection = _FailAfterHandoffStarts(
        _welcome("session-old-cancel"),
        _reconnect("reconnect-cancel", reconnect_url),
    )
    connector = _FakeConnector(old_connection, blocked_handoff)
    source = _source(_FakeHttpClient(), connector)
    source.start()

    _wait_until(lambda: blocked_handoff.blocked_in_recv.is_set())
    _wait_until(lambda: blocked_handoff.closed.is_set())
    with source._connections_lock:
        assert all(connection is not blocked_handoff for connection in source._connections)
    assert source.stop(timeout_seconds=1.0)


def test_ordinary_disconnect_reconnects_and_creates_a_new_subscription() -> None:
    old_connection = _FakeWebSocket(
        _welcome("session-a"),
        _notification("delivery-a", "event-a"),
        ConnectionError("socket_lost"),
    )
    new_connection = _FakeWebSocket(
        _welcome("session-b"),
        _notification("delivery-b", "event-b"),
    )
    http = _FakeHttpClient()
    connector = _FakeConnector(old_connection, new_connection)
    source = _source(http, connector)
    source.start()
    _wait_until(lambda: source.get_diagnostics()["twitch_notifications_enqueued"] >= 2)
    assert http.subscription_sessions == [
        ("session-a", READER_ID),
        ("session-b", READER_ID),
    ]
    assert source.get_diagnostics()["twitch_ordinary_reconnects"] >= 1
    assert source.get_diagnostics()["twitch_network_available"] is True
    assert [message.event_id for message in source.drain_messages()] == ["event-b"]
    assert old_connection.closed.is_set()
    assert source.stop(timeout_seconds=1.0)


def test_immediate_post_subscription_disconnects_use_capped_exponential_backoff() -> None:
    import asyncio

    delays: list[float] = []
    connector = _FakeConnector(
        *(
            _FakeWebSocket(
                _welcome(f"unstable-{index}"),
                ConnectionError("immediate_disconnect"),
            )
            for index in range(4)
        )
    )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 4:
            source._stop_requested.set()

    source = TwitchEventSubSource(
        _credentials(),
        http_client=_FakeHttpClient(),
        websocket_connector=connector,
        backoff_base_seconds=1.0,
        backoff_max_seconds=4.0,
        random_value=lambda: 0.5,
        async_sleep=record_sleep,
    )
    asyncio.run(source._run())

    assert delays == [1.0, 2.0, 4.0, 4.0]
    diagnostics = source.get_diagnostics()
    assert diagnostics["twitch_reconnect_attempt"] == 4
    assert diagnostics["twitch_last_backoff_seconds"] == 4.0


def test_valid_session_activity_resets_ordinary_reconnect_backoff() -> None:
    import asyncio

    delays: list[float] = []
    connector = _FakeConnector(
        _FakeWebSocket(_welcome("first"), ConnectionError("first_disconnect")),
        _FakeWebSocket(
            _welcome("stable"),
            _notification("stable-delivery", "stable-event"),
            ConnectionError("stable_disconnect"),
        ),
        _FakeWebSocket(_welcome("third"), ConnectionError("third_disconnect")),
    )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 3:
            source._stop_requested.set()

    source = TwitchEventSubSource(
        _credentials(),
        http_client=_FakeHttpClient(),
        websocket_connector=connector,
        backoff_base_seconds=1.0,
        backoff_max_seconds=8.0,
        random_value=lambda: 0.5,
        async_sleep=record_sleep,
    )
    asyncio.run(source._run())

    assert delays == [1.0, 1.0, 2.0]


def test_transient_startup_validation_failure_retries_without_exposing_secrets() -> None:
    validation = TokenValidation(
        client_id="client-7",
        user_id=READER_ID,
        scopes=("user:read:chat",),
        expires_in=14_400,
    )
    http = _FakeHttpClient(TwitchHttpError(0, "network_timeout"), validation)
    connection = _FakeWebSocket(_welcome("session-retry"))
    connector = _FakeConnector(connection)
    source = _source(http, connector)
    source.start()
    _wait_until(lambda: source.get_diagnostics()["twitch_connection_state"] == "connected")
    diagnostics_text = json.dumps(source.get_diagnostics())
    assert http.validate_calls == 2
    assert source.get_diagnostics()["twitch_ordinary_reconnects"] >= 1
    assert TOKEN not in diagnostics_text
    assert HASH_SECRET.decode("utf-8") not in diagnostics_text
    assert source.stop(timeout_seconds=1.0)


@pytest.mark.parametrize(
    ("frame", "expected_state"),
    [
        (_revocation("revoke-1", "authorization_revoked"), "auth_failed"),
    ],
)
def test_revocation_fails_closed_and_clears_pending_messages(frame: str, expected_state: str) -> None:
    connection = _FakeWebSocket(
        _welcome("session-revoked"),
        _notification("delivery-before-revoke", "event-before-revoke"),
        frame,
    )
    source = _source(_FakeHttpClient(), _FakeConnector(connection))
    source.start()
    _wait_until(lambda: source.get_diagnostics()["twitch_connection_state"] == expected_state)
    diagnostics = source.get_diagnostics()
    assert diagnostics["stream_source_queue_depth"] == 0
    assert diagnostics["twitch_revocations"] == 1
    assert source.stop(timeout_seconds=1.0)


def test_invalid_token_is_reported_safely_without_opening_websocket() -> None:
    http = _FakeHttpClient(TwitchHttpError(401, "invalid_access_token"))
    connector = _FakeConnector()
    source = _source(http, connector)
    source.start()
    _wait_until(lambda: source.get_diagnostics()["twitch_connection_state"] == "auth_failed")
    diagnostics_text = json.dumps(source.get_diagnostics())
    assert connector.urls == []
    assert "invalid_access_token" in diagnostics_text
    assert TOKEN not in diagnostics_text
    assert HASH_SECRET.decode("utf-8") not in diagnostics_text
    assert source.stop(timeout_seconds=1.0)


def test_keepalive_watchdog_and_blocked_receive_shutdown_are_bounded() -> None:
    watchdog = EventSubSessionWatchdog(10.0, observed_at=5.0)
    assert watchdog.remaining(7.5) == pytest.approx(7.5)
    assert not watchdog.expired(15.0)
    assert watchdog.expired(15.001)
    watchdog.observe(20.0)
    assert watchdog.remaining(20.0) == pytest.approx(10.0)

    connection = _FakeWebSocket(_welcome("session-blocked"))
    source = _source(_FakeHttpClient(), _FakeConnector(connection))
    source.start()
    _wait_until(lambda: connection.blocked_in_recv.is_set())
    started = time.monotonic()
    assert source.stop(timeout_seconds=1.0)
    assert time.monotonic() - started < 1.0
    assert connection.closed.is_set()
