"""Twitch EventSub WebSocket source for local PvZRL Streamer Mode.

The adapter is deliberately source-only: it normalizes Twitch chat events into
privacy-safe ``StreamSourceMessage`` records and never calls the game, bridge,
or command parser.  All networking lives on one bounded background worker.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import random
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from pvzrl_streamer_source import BoundedStreamMessageBuffer, StreamSourceMessage


TWITCH_EVENTSUB_URL = "wss://eventsub.wss.twitch.tv/ws?keepalive_timeout_seconds=30"
TWITCH_SUBSCRIPTIONS_URL = "https://api.twitch.tv/helix/eventsub/subscriptions"
TWITCH_VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
TWITCH_SUBSCRIPTION_TYPE = "channel.chat.message"
TWITCH_SUBSCRIPTION_VERSION = "1"

TWITCH_CLIENT_ID_ENV = "PVZRL_TWITCH_CLIENT_ID"
TWITCH_ACCESS_TOKEN_ENV = "PVZRL_TWITCH_USER_ACCESS_TOKEN"
TWITCH_BROADCASTER_ID_ENV = "PVZRL_TWITCH_BROADCASTER_USER_ID"
TWITCH_EVENTSUB_USER_ID_ENV = "PVZRL_TWITCH_EVENTSUB_USER_ID"
TWITCH_VIEWER_HASH_SECRET_ENV = "PVZRL_TWITCH_VIEWER_HASH_SECRET"


def _diagnostic_code(value: Any, fallback: str) -> str:
    raw = value if isinstance(value, str) else ""
    normalized = "".join(
        character if character.isascii() and (character.isalnum() or character in "_.-") else "_"
        for character in raw
    ).strip("_.-")
    return (normalized or fallback)[:128]


class TwitchConfigurationError(ValueError):
    """Safe configuration failure containing names, never credential values."""


class TwitchHttpError(RuntimeError):
    """Redacted Twitch HTTP failure."""

    def __init__(self, status: int, code: str, *, existing_subscription_id: str = "") -> None:
        self.status = int(status)
        self.code = _diagnostic_code(code, "http_error")
        self.existing_subscription_id = str(existing_subscription_id or "")
        super().__init__(f"{self.code} (HTTP {self.status})")


class EventSubProtocolError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = _diagnostic_code(code, "eventsub_protocol_error")
        super().__init__(self.code)


@dataclass(frozen=True)
class TwitchCredentials:
    """Credentials and identities required by the read-only Twitch adapter."""

    client_id: str = field(repr=False)
    access_token: str = field(repr=False)
    broadcaster_user_id: str = field(repr=False)
    viewer_hash_secret: bytes = field(repr=False)
    eventsub_user_id: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        def normalize(
            value: Any,
            env_name: str,
            *,
            optional: bool = False,
            max_length: int = 256,
        ) -> str:
            if not isinstance(value, str):
                raise TwitchConfigurationError(f"invalid {env_name}")
            normalized = value.strip()
            if not normalized:
                if optional:
                    return ""
                raise TwitchConfigurationError(f"missing {env_name}")
            if len(normalized) > max_length or any(
                not 33 <= ord(character) <= 126 for character in normalized
            ):
                raise TwitchConfigurationError(f"invalid {env_name}")
            return normalized

        object.__setattr__(self, "client_id", normalize(self.client_id, TWITCH_CLIENT_ID_ENV))
        object.__setattr__(
            self,
            "access_token",
            normalize(self.access_token, TWITCH_ACCESS_TOKEN_ENV, max_length=4_096),
        )
        object.__setattr__(
            self,
            "broadcaster_user_id",
            normalize(self.broadcaster_user_id, TWITCH_BROADCASTER_ID_ENV),
        )
        object.__setattr__(
            self,
            "eventsub_user_id",
            normalize(self.eventsub_user_id, TWITCH_EVENTSUB_USER_ID_ENV, optional=True),
        )
        if not isinstance(self.viewer_hash_secret, (bytes, bytearray)):
            raise TwitchConfigurationError(f"invalid {TWITCH_VIEWER_HASH_SECRET_ENV}")
        secret = bytes(self.viewer_hash_secret)
        if not 16 <= len(secret) <= 4_096:
            raise TwitchConfigurationError(
                f"{TWITCH_VIEWER_HASH_SECRET_ENV} must contain between 16 and 4096 bytes"
            )
        object.__setattr__(self, "viewer_hash_secret", secret)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "TwitchCredentials":
        values = os.environ if env is None else env
        secret = str(values.get(TWITCH_VIEWER_HASH_SECRET_ENV, "") or "")
        return cls(
            client_id=str(values.get(TWITCH_CLIENT_ID_ENV, "") or "").strip(),
            access_token=str(values.get(TWITCH_ACCESS_TOKEN_ENV, "") or "").strip(),
            broadcaster_user_id=str(values.get(TWITCH_BROADCASTER_ID_ENV, "") or "").strip(),
            eventsub_user_id=str(values.get(TWITCH_EVENTSUB_USER_ID_ENV, "") or "").strip(),
            viewer_hash_secret=secret.encode("utf-8"),
        )


@dataclass(frozen=True)
class TokenValidation:
    client_id: str = field(repr=False)
    user_id: str = field(repr=False)
    scopes: Tuple[str, ...]
    expires_in: int


@dataclass(frozen=True)
class SubscriptionResult:
    subscription_id: str
    status: str
    session_id: str


class TwitchEventSubHttpClient(Protocol):
    def validate_token(self, credentials: TwitchCredentials) -> TokenValidation:
        ...

    def create_chat_subscription(
        self,
        credentials: TwitchCredentials,
        *,
        session_id: str,
        eventsub_user_id: str,
    ) -> SubscriptionResult:
        ...


class EventSubWebSocket(Protocol):
    async def recv(self) -> Any:
        ...

    async def close(self) -> Any:
        ...


WebSocketConnector = Callable[[str], Awaitable[EventSubWebSocket]]


def _http_error_code(status: int) -> str:
    return {
        400: "bad_request",
        401: "invalid_access_token",
        403: "missing_authorization",
        409: "subscription_conflict",
        410: "subscription_version_removed",
        429: "rate_limited",
    }.get(int(status), f"http_{int(status)}")


class UrllibTwitchEventSubHttpClient:
    """Small HTTPS client whose errors never expose headers or response text."""

    def __init__(self, *, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = max(0.25, float(timeout_seconds))
        self._ssl_context = ssl.create_default_context()

    def validate_token(self, credentials: TwitchCredentials) -> TokenValidation:
        request = urllib.request.Request(
            TWITCH_VALIDATE_URL,
            method="GET",
            headers={"Authorization": f"OAuth {credentials.access_token}"},
        )
        payload = self._request_json(request, expected_status=200)
        scopes = payload.get("scopes")
        if not isinstance(scopes, list):
            raise TwitchHttpError(200, "token_validation_response_invalid")
        try:
            expires_in = int(payload.get("expires_in"))
        except (TypeError, ValueError) as exc:
            raise TwitchHttpError(200, "token_validation_response_invalid") from exc
        client_id = payload.get("client_id")
        user_id = payload.get("user_id")
        if not isinstance(client_id, str) or not client_id or not isinstance(user_id, str) or not user_id:
            raise TwitchHttpError(200, "token_validation_response_invalid")
        return TokenValidation(
            client_id=client_id,
            user_id=user_id,
            scopes=tuple(str(scope) for scope in scopes),
            expires_in=expires_in,
        )

    def create_chat_subscription(
        self,
        credentials: TwitchCredentials,
        *,
        session_id: str,
        eventsub_user_id: str,
    ) -> SubscriptionResult:
        body = json.dumps(
            {
                "type": TWITCH_SUBSCRIPTION_TYPE,
                "version": TWITCH_SUBSCRIPTION_VERSION,
                "condition": {
                    "broadcaster_user_id": credentials.broadcaster_user_id,
                    "user_id": eventsub_user_id,
                },
                "transport": {"method": "websocket", "session_id": session_id},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            TWITCH_SUBSCRIPTIONS_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {credentials.access_token}",
                "Client-Id": credentials.client_id,
                "Content-Type": "application/json",
            },
        )
        payload = self._request_json(request, expected_status=202)
        data = payload.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise TwitchHttpError(202, "subscription_response_invalid")
        item = data[0]
        transport = item.get("transport")
        returned_session_id = transport.get("session_id") if isinstance(transport, dict) else ""
        subscription_id = item.get("id")
        status = item.get("status")
        if not all(isinstance(value, str) and value for value in (subscription_id, status, returned_session_id)):
            raise TwitchHttpError(202, "subscription_response_invalid")
        return SubscriptionResult(
            subscription_id=subscription_id,
            status=status,
            session_id=returned_session_id,
        )

    def _request_json(self, request: urllib.request.Request, *, expected_status: int) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self._ssl_context,
            ) as response:
                status = int(getattr(response, "status", response.getcode()))
                raw = response.read(1_048_577)
        except urllib.error.HTTPError as exc:
            existing_id = ""
            if int(exc.code) == 409:
                try:
                    raw_error = exc.read(65_537)
                    error_payload = json.loads(raw_error.decode("utf-8"))
                    candidate = error_payload.get("id") if isinstance(error_payload, dict) else None
                    existing_id = str(candidate or "")[:128]
                except (OSError, UnicodeError, json.JSONDecodeError):
                    existing_id = ""
            raise TwitchHttpError(
                int(exc.code),
                _http_error_code(int(exc.code)),
                existing_subscription_id=existing_id,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TwitchHttpError(0, f"network_{type(exc).__name__.lower()}") from None
        if status != int(expected_status):
            raise TwitchHttpError(status, _http_error_code(status))
        if len(raw) > 1_048_576:
            raise TwitchHttpError(status, "http_response_too_large")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise TwitchHttpError(status, "http_response_invalid_json") from exc
        if not isinstance(payload, dict):
            raise TwitchHttpError(status, "http_response_invalid_json")
        return payload


class BoundedTtlDeduplicator:
    def __init__(self, *, capacity: int, ttl_seconds: float) -> None:
        if int(capacity) <= 0:
            raise ValueError("dedupe capacity must be positive")
        if float(ttl_seconds) <= 0.0:
            raise ValueError("dedupe TTL must be positive")
        self.capacity = int(capacity)
        self.ttl_seconds = float(ttl_seconds)
        self._entries: "OrderedDict[str, float]" = OrderedDict()

    def seen_or_add(self, key: str, *, now: float) -> bool:
        self._prune(float(now))
        normalized = str(key)
        if normalized in self._entries:
            self._entries.move_to_end(normalized)
            return True
        self._entries[normalized] = float(now)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return False

    def _prune(self, now: float) -> None:
        cutoff = float(now) - self.ttl_seconds
        while self._entries:
            _, seen_at = next(iter(self._entries.items()))
            if seen_at >= cutoff:
                break
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class EventSubWelcome:
    session_id: str
    keepalive_timeout_seconds: float


@dataclass(frozen=True)
class EventSubFrameResult:
    kind: str
    welcome: Optional[EventSubWelcome] = None
    message: Optional[StreamSourceMessage] = None
    reconnect_url: str = ""
    revocation_status: str = ""
    error_code: str = ""
    delivery_id: str = ""
    event_id: str = ""


class EventSubSessionWatchdog:
    """Monotonic EventSub silence watchdog (WebSocket Ping does not reset it)."""

    def __init__(self, timeout_seconds: float, *, observed_at: float) -> None:
        if float(timeout_seconds) <= 0.0:
            raise ValueError("keepalive timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.last_observed = float(observed_at)

    def observe(self, observed_at: float) -> None:
        self.last_observed = float(observed_at)

    def remaining(self, now: float) -> float:
        return max(0.0, self.timeout_seconds - (float(now) - self.last_observed))

    def expired(self, now: float) -> bool:
        return float(now) - self.last_observed > self.timeout_seconds


class TwitchEventSubProtocol:
    """Pure EventSub frame validation, normalization, privacy, and dedupe."""

    def __init__(
        self,
        credentials: TwitchCredentials,
        *,
        max_frame_bytes: int = 262_144,
        max_command_chars: int = 512,
        max_command_bytes: int = 2_048,
        dedupe_capacity: int = 10_000,
        dedupe_ttl_seconds: float = 900.0,
    ) -> None:
        self.credentials = credentials
        self.max_frame_bytes = max(1_024, int(max_frame_bytes))
        self.max_command_chars = max(1, int(max_command_chars))
        self.max_command_bytes = max(1, int(max_command_bytes))
        self._delivery_ids = BoundedTtlDeduplicator(
            capacity=dedupe_capacity,
            ttl_seconds=dedupe_ttl_seconds,
        )
        self._chat_ids = BoundedTtlDeduplicator(
            capacity=dedupe_capacity,
            ttl_seconds=dedupe_ttl_seconds,
        )
        self._control_ids = BoundedTtlDeduplicator(
            capacity=max(128, min(dedupe_capacity, 1_024)),
            ttl_seconds=dedupe_ttl_seconds,
        )
        self._eventsub_user_id = str(credentials.eventsub_user_id or "")

    def set_eventsub_user_id(self, eventsub_user_id: str) -> None:
        if not self._valid_identifier(eventsub_user_id):
            raise TwitchConfigurationError(f"invalid {TWITCH_EVENTSUB_USER_ID_ENV}")
        self._eventsub_user_id = str(eventsub_user_id)

    def parse(self, raw_frame: Any, *, received_monotonic: float) -> EventSubFrameResult:
        raw_text = self._decode_frame(raw_frame)
        if raw_text is None:
            return EventSubFrameResult(kind="malformed", error_code="frame_invalid_encoding_or_size")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return EventSubFrameResult(kind="malformed", error_code="invalid_json")
        if not isinstance(payload, dict):
            return EventSubFrameResult(kind="malformed", error_code="payload_not_object")
        metadata = payload.get("metadata")
        body = payload.get("payload")
        if not isinstance(metadata, dict) or not isinstance(body, dict):
            return EventSubFrameResult(kind="malformed", error_code="missing_metadata_or_payload")
        message_type = metadata.get("message_type")
        delivery_id = metadata.get("message_id")
        if not isinstance(message_type, str) or not self._valid_identifier(delivery_id):
            return EventSubFrameResult(kind="malformed", error_code="invalid_message_metadata")
        delivery_id = str(delivery_id)

        if message_type == "session_welcome":
            session = body.get("session")
            if (
                not isinstance(session, dict)
                or not self._valid_identifier(session.get("id"))
                or session.get("status") != "connected"
            ):
                return EventSubFrameResult(kind="malformed", error_code="invalid_welcome")
            timeout = session.get("keepalive_timeout_seconds")
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not 0.0 < float(timeout) <= 600.0
            ):
                return EventSubFrameResult(kind="malformed", error_code="invalid_welcome_timeout")
            return EventSubFrameResult(
                kind="welcome",
                welcome=EventSubWelcome(
                    session_id=str(session["id"]),
                    keepalive_timeout_seconds=float(timeout),
                ),
                delivery_id=delivery_id,
            )

        if message_type == "session_keepalive":
            return EventSubFrameResult(kind="keepalive", delivery_id=delivery_id)

        if message_type == "session_reconnect":
            if self._control_ids.seen_or_add(f"reconnect:{delivery_id}", now=received_monotonic):
                return EventSubFrameResult(kind="duplicate", error_code="duplicate_reconnect", delivery_id=delivery_id)
            session = body.get("session")
            reconnect_url = session.get("reconnect_url") if isinstance(session, dict) else None
            if not self._valid_reconnect_url(reconnect_url):
                return EventSubFrameResult(kind="malformed", error_code="invalid_reconnect_url")
            return EventSubFrameResult(
                kind="reconnect",
                reconnect_url=str(reconnect_url),
                delivery_id=delivery_id,
            )

        if message_type == "revocation":
            if self._control_ids.seen_or_add(f"revocation:{delivery_id}", now=received_monotonic):
                return EventSubFrameResult(
                    kind="duplicate",
                    error_code="duplicate_revocation",
                    delivery_id=delivery_id,
                )
            subscription = body.get("subscription")
            status = subscription.get("status") if isinstance(subscription, dict) else None
            if not self._valid_status(status):
                return EventSubFrameResult(kind="malformed", error_code="invalid_revocation")
            return EventSubFrameResult(
                kind="revocation",
                revocation_status=status,
                delivery_id=delivery_id,
            )

        if message_type != "notification":
            return EventSubFrameResult(kind="ignored", error_code="unknown_message_type", delivery_id=delivery_id)
        return self._parse_notification(
            metadata,
            body,
            delivery_id=delivery_id,
            received_monotonic=float(received_monotonic),
        )

    def _parse_notification(
        self,
        metadata: Dict[str, Any],
        body: Dict[str, Any],
        *,
        delivery_id: str,
        received_monotonic: float,
    ) -> EventSubFrameResult:
        if metadata.get("subscription_type") != TWITCH_SUBSCRIPTION_TYPE:
            return EventSubFrameResult(kind="ignored", error_code="unexpected_subscription_type")
        if str(metadata.get("subscription_version") or "") != TWITCH_SUBSCRIPTION_VERSION:
            return EventSubFrameResult(kind="ignored", error_code="unexpected_subscription_version")
        subscription = body.get("subscription")
        event = body.get("event")
        if not isinstance(subscription, dict) or not isinstance(event, dict):
            return EventSubFrameResult(kind="malformed", error_code="invalid_notification_payload")
        condition = subscription.get("condition")
        if (
            subscription.get("type") != TWITCH_SUBSCRIPTION_TYPE
            or str(subscription.get("version") or "") != TWITCH_SUBSCRIPTION_VERSION
        ):
            return EventSubFrameResult(kind="ignored", error_code="subscription_payload_mismatch")
        if (
            not isinstance(condition, dict)
            or condition.get("broadcaster_user_id") != self.credentials.broadcaster_user_id
        ):
            return EventSubFrameResult(kind="ignored", error_code="subscription_condition_mismatch")
        if self._eventsub_user_id and condition.get("user_id") != self._eventsub_user_id:
            return EventSubFrameResult(kind="ignored", error_code="subscription_user_mismatch")
        if event.get("broadcaster_user_id") != self.credentials.broadcaster_user_id:
            return EventSubFrameResult(kind="ignored", error_code="event_broadcaster_mismatch")
        event_id = event.get("message_id")
        chatter_user_id = event.get("chatter_user_id")
        message = event.get("message")
        command_text = message.get("text") if isinstance(message, dict) else None
        if not self._valid_identifier(event_id) or not self._valid_identifier(chatter_user_id):
            return EventSubFrameResult(kind="malformed", error_code="invalid_chat_identity")
        if not isinstance(command_text, str):
            return EventSubFrameResult(kind="malformed", error_code="invalid_chat_text")
        if len(command_text) > self.max_command_chars:
            return EventSubFrameResult(kind="malformed", error_code="chat_text_too_long")
        try:
            encoded_text = command_text.encode("utf-8")
        except UnicodeEncodeError:
            return EventSubFrameResult(kind="malformed", error_code="chat_text_invalid_unicode")
        if len(encoded_text) > self.max_command_bytes:
            return EventSubFrameResult(kind="malformed", error_code="chat_text_too_long")
        first_token = command_text.lstrip().split(maxsplit=1)[0].casefold() if command_text.strip() else ""
        if first_token not in {"!plant", "!slot", "!fuse"}:
            # Ordinary channel conversation is not a rejected command and never
            # enters the hot-path buffer or privacy-sensitive identity hashing.
            return EventSubFrameResult(kind="ignored", error_code="non_command_chat")
        if self._delivery_ids.seen_or_add(f"delivery:{delivery_id}", now=received_monotonic):
            return EventSubFrameResult(
                kind="duplicate",
                error_code="duplicate_delivery",
                delivery_id=delivery_id,
                event_id=str(event_id),
            )
        chat_key = f"chat:{self.credentials.broadcaster_user_id}:{event_id}"
        if self._chat_ids.seen_or_add(chat_key, now=received_monotonic):
            return EventSubFrameResult(
                kind="duplicate",
                error_code="duplicate_chat_event",
                delivery_id=delivery_id,
                event_id=str(event_id),
            )
        viewer_hash = hmac.new(
            self.credentials.viewer_hash_secret,
            str(chatter_user_id).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        published_at = metadata.get("message_timestamp")
        if (
            not isinstance(published_at, str)
            or len(published_at) > 64
            or any(not 33 <= ord(character) <= 126 for character in published_at)
        ):
            published_at = None
        return EventSubFrameResult(
            kind="notification",
            delivery_id=delivery_id,
            event_id=str(event_id),
            message=StreamSourceMessage(
                platform="twitch",
                delivery_id=delivery_id,
                event_id=str(event_id),
                viewer_hash=viewer_hash,
                command_text=command_text,
                received_monotonic=float(received_monotonic),
                published_at=published_at,
            ),
        )

    def dedupe_size(self) -> int:
        return len(self._delivery_ids) + len(self._chat_ids) + len(self._control_ids)

    def _decode_frame(self, raw_frame: Any) -> Optional[str]:
        if isinstance(raw_frame, bytes):
            if len(raw_frame) > self.max_frame_bytes:
                return None
            try:
                return raw_frame.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(raw_frame, str):
            return None
        try:
            encoded = raw_frame.encode("utf-8")
        except UnicodeEncodeError:
            return None
        return raw_frame if len(encoded) <= self.max_frame_bytes else None

    @staticmethod
    def _valid_identifier(value: Any) -> bool:
        return (
            isinstance(value, str)
            and 0 < len(value) <= 256
            and all(33 <= ord(character) <= 126 for character in value)
        )

    @staticmethod
    def _valid_status(value: Any) -> bool:
        return (
            isinstance(value, str)
            and 0 < len(value) <= 128
            and all(
                "a" <= character <= "z" or "0" <= character <= "9" or character == "_"
                for character in value
            )
        )

    @staticmethod
    def _valid_reconnect_url(value: Any) -> bool:
        if not isinstance(value, str) or len(value) > 2_048:
            return False
        try:
            parsed = urllib.parse.urlsplit(value)
            return (
                parsed.scheme == "wss"
                and parsed.hostname == "eventsub.wss.twitch.tv"
                and parsed.port in {None, 443}
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            return False


@dataclass(frozen=True)
class _ConsumeOutcome:
    kind: str
    error_code: str = ""
    connection: Optional[EventSubWebSocket] = None
    welcome: Optional[EventSubWelcome] = None


class TwitchEventSubSource:
    """Background EventSub producer implementing ``StreamCommandSource``."""

    def __init__(
        self,
        credentials: Optional[TwitchCredentials] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        http_client: Optional[TwitchEventSubHttpClient] = None,
        websocket_connector: Optional[WebSocketConnector] = None,
        queue_capacity: int = 256,
        accepting: bool = True,
        max_frame_bytes: int = 262_144,
        max_command_chars: int = 512,
        max_command_bytes: int = 2_048,
        dedupe_capacity: int = 10_000,
        dedupe_ttl_seconds: float = 900.0,
        welcome_timeout_seconds: float = 10.0,
        open_timeout_seconds: float = 10.0,
        backoff_base_seconds: float = 0.5,
        backoff_max_seconds: float = 30.0,
        token_validation_interval_seconds: float = 3_600.0,
        monotonic: Callable[[], float] = time.monotonic,
        random_value: Callable[[], float] = random.random,
        async_sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._credentials = credentials if credentials is not None else TwitchCredentials.from_env(env)
        self._http = http_client if http_client is not None else UrllibTwitchEventSubHttpClient()
        self._max_frame_bytes = max(1_024, int(max_frame_bytes))
        self._websocket_connector = websocket_connector or self._default_websocket_connect
        # Network availability and the runtime TRAIN/EVALUATE phase are
        # independent gates.  The effective queue gate opens only when both do.
        self._buffer = BoundedStreamMessageBuffer(queue_capacity, accepting=False)
        self._protocol = TwitchEventSubProtocol(
            self._credentials,
            max_frame_bytes=self._max_frame_bytes,
            max_command_chars=max_command_chars,
            max_command_bytes=max_command_bytes,
            dedupe_capacity=dedupe_capacity,
            dedupe_ttl_seconds=dedupe_ttl_seconds,
        )
        self._welcome_timeout_seconds = max(0.25, float(welcome_timeout_seconds))
        self._open_timeout_seconds = max(0.25, float(open_timeout_seconds))
        self._backoff_base_seconds = max(0.0, float(backoff_base_seconds))
        self._backoff_max_seconds = max(self._backoff_base_seconds, float(backoff_max_seconds))
        self._token_validation_interval_seconds = max(60.0, float(token_validation_interval_seconds))
        self._monotonic = monotonic
        self._random_value = random_value
        self._async_sleep = async_sleep

        self._lifecycle_lock = threading.Lock()
        self._phase_lock = threading.Lock()
        self._diagnostics_lock = threading.Lock()
        self._connections_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task[Any]] = None
        self._connections: List[EventSubWebSocket] = []
        self._reader_user_id = ""
        self._next_token_validation = 0.0
        self._connection_activity_count = 0
        self._phase_accepting = bool(accepting)
        self._network_available = False
        self._stop_requested = threading.Event()
        self._counts: Counter[str] = Counter()
        self._diagnostics: Dict[str, Any] = {
            "twitch_connection_state": "stopped",
            "twitch_last_error": "",
            "twitch_last_revocation_status": "",
            "twitch_session_id": "",
            "twitch_subscription_id": "",
            "twitch_keepalive_timeout_seconds": 0.0,
            "twitch_last_delivery_id": "",
            "twitch_last_event_id": "",
            "twitch_token_expires_in": 0,
            "twitch_worker_alive": False,
            "twitch_shutdown_timed_out": False,
            "twitch_reconnect_attempt": 0,
            "twitch_last_backoff_seconds": 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(broadcaster_configured=True, "
            f"queue_capacity={self._buffer.capacity})"
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested.clear()
            self._set_diagnostics(
                twitch_connection_state="starting",
                twitch_last_error="",
                twitch_last_revocation_status="",
                twitch_session_id="",
                twitch_subscription_id="",
                twitch_keepalive_timeout_seconds=0.0,
                twitch_last_delivery_id="",
                twitch_last_event_id="",
                twitch_shutdown_timed_out=False,
                twitch_reconnect_attempt=0,
                twitch_last_backoff_seconds=0.0,
            )
            self._thread = threading.Thread(
                target=self._thread_main,
                name="pvzrl-twitch-eventsub",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_seconds: float = 5.0) -> bool:
        timeout = max(0.0, float(timeout_seconds))
        self._stop_requested.set()
        self._set_network_available(False, reason="source_stopped", force_epoch=True)
        self._set_diagnostics(twitch_connection_state="stopping")
        with self._lifecycle_lock:
            loop = self._loop
            task = self._main_task
            thread = self._thread
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        alive = bool(thread is not None and thread.is_alive())
        self._set_diagnostics(
            twitch_worker_alive=alive,
            twitch_shutdown_timed_out=alive,
            twitch_connection_state="shutdown_timeout" if alive else "stopped",
        )
        return not alive

    def drain_messages(self, max_items: Optional[int] = None) -> List[StreamSourceMessage]:
        return self._buffer.drain(max_items=max_items)

    def clear(self) -> int:
        return self._buffer.clear()

    def set_accepting(self, accepting: bool, *, reason: str = "phase_change") -> int:
        with self._phase_lock:
            self._phase_accepting = bool(accepting)
            effective_accepting = self._phase_accepting and self._network_available
            return self._buffer.set_accepting(effective_accepting, reason=reason)

    def get_diagnostics(self) -> Dict[str, Any]:
        with self._phase_lock:
            phase_accepting = bool(self._phase_accepting)
            network_available = bool(self._network_available)
        with self._diagnostics_lock:
            diagnostics = dict(self._diagnostics)
            diagnostics.update(
                {
                    "twitch_phase_accepting": phase_accepting,
                    "twitch_network_available": network_available,
                    "twitch_notifications_received": int(self._counts["notifications_received"]),
                    "twitch_notifications_enqueued": int(self._counts["notifications_enqueued"]),
                    "twitch_duplicate_events": int(self._counts["duplicate_events"]),
                    "twitch_malformed_events": int(self._counts["malformed_events"]),
                    "twitch_ignored_events": int(self._counts["ignored_events"]),
                    "twitch_keepalives_received": int(self._counts["keepalives_received"]),
                    "twitch_keepalive_timeouts": int(self._counts["keepalive_timeouts"]),
                    "twitch_reconnect_requests": int(self._counts["reconnect_requests"]),
                    "twitch_reconnect_handoffs": int(self._counts["reconnect_handoffs"]),
                    "twitch_ordinary_reconnects": int(self._counts["ordinary_reconnects"]),
                    "twitch_subscriptions_created": int(self._counts["subscriptions_created"]),
                    "twitch_token_validations": int(self._counts["token_validations"]),
                    "twitch_revocations": int(self._counts["revocations"]),
                    "twitch_dedupe_cache_size": int(self._protocol.dedupe_size()),
                }
            )
        diagnostics.update(self._buffer.diagnostics())
        return diagnostics

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._run())
        with self._lifecycle_lock:
            self._loop = loop
            self._main_task = task
        self._set_diagnostics(twitch_worker_alive=True)
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._set_diagnostics(
                twitch_connection_state="failed",
                twitch_last_error=_safe_exception_code(exc),
            )
        finally:
            try:
                try:
                    loop.run_until_complete(self._close_all_connections())
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self._set_diagnostics(twitch_last_error=_safe_exception_code(exc))
            finally:
                with self._lifecycle_lock:
                    self._main_task = None
                    self._loop = None
                self._set_diagnostics(twitch_worker_alive=False)
                loop.close()

    async def _run(self) -> None:
        connection: Optional[EventSubWebSocket] = None
        reconnect_attempt = 0
        try:
            while not self._stop_requested.is_set():
                try:
                    if not self._reader_user_id or self._monotonic() >= self._next_token_validation:
                        self._validate_token()
                    self._set_diagnostics(twitch_connection_state="connecting")
                    connection, welcome = await self._connect_and_welcome(TWITCH_EVENTSUB_URL)
                    self._set_diagnostics(twitch_connection_state="subscribing")
                    subscription = self._http.create_chat_subscription(
                        self._credentials,
                        session_id=welcome.session_id,
                        eventsub_user_id=self._reader_user_id,
                    )
                    if subscription.status != "enabled" or subscription.session_id != welcome.session_id:
                        raise TwitchHttpError(202, "subscription_not_enabled")
                    if self._stop_requested.is_set():
                        raise asyncio.CancelledError
                    self._increment("subscriptions_created")
                    self._set_network_available(True, reason="twitch_connected")
                    self._set_diagnostics(
                        twitch_connection_state="connected",
                        twitch_subscription_id=str(subscription.subscription_id)[:256],
                        twitch_last_error="",
                    )
                    # A welcome/subscription alone is not proof of a stable
                    # session.  Otherwise a server that repeatedly accepts a
                    # subscription and immediately disconnects defeats the
                    # exponential backoff.  Reset only after a valid live
                    # session frame is observed by ``_consume_connection``.
                    activity_count_at_connect = self._connection_activity_count

                    while not self._stop_requested.is_set():
                        outcome = await self._consume_connection(connection, welcome)
                        if self._connection_activity_count != activity_count_at_connect:
                            reconnect_attempt = 0
                            activity_count_at_connect = self._connection_activity_count
                            self._set_diagnostics(
                                twitch_reconnect_attempt=0,
                                twitch_last_backoff_seconds=0.0,
                            )
                        if (
                            outcome.kind == "handoff"
                            and outcome.connection is not None
                            and outcome.welcome is not None
                        ):
                            connection = outcome.connection
                            welcome = outcome.welcome
                            self._set_diagnostics(twitch_connection_state="connected", twitch_last_error="")
                            continue
                        if outcome.kind == "auth_failed":
                            self._buffer.clear()
                            self._set_diagnostics(
                                twitch_connection_state="auth_failed",
                                twitch_last_error=outcome.error_code or "authorization_revoked",
                            )
                            return
                        if outcome.kind == "revoked":
                            self._buffer.clear()
                            self._set_diagnostics(
                                twitch_connection_state="revoked",
                                twitch_last_error=outcome.error_code or "subscription_revoked",
                            )
                            return
                        self._buffer.clear()
                        raise EventSubProtocolError(outcome.error_code or "ordinary_disconnect")
                except TwitchHttpError as exc:
                    code = _safe_exception_code(exc)
                    if exc.status in {401, 403}:
                        self._buffer.clear()
                        self._set_diagnostics(twitch_connection_state="auth_failed", twitch_last_error=code)
                        return
                    if exc.status == 410:
                        self._buffer.clear()
                        self._set_diagnostics(twitch_connection_state="revoked", twitch_last_error=code)
                        return
                    if exc.status == 400:
                        self._buffer.clear()
                        self._set_diagnostics(
                            twitch_connection_state="configuration_error",
                            twitch_last_error=code,
                        )
                        return
                    self._buffer.clear()
                    self._set_diagnostics(twitch_last_error=code)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._buffer.clear()
                    self._set_diagnostics(twitch_last_error=_safe_exception_code(exc))
                finally:
                    self._set_network_available(False, reason="twitch_disconnected")
                    if connection is not None:
                        await self._close_connection(connection)
                        connection = None

                if self._stop_requested.is_set():
                    break
                self._increment("ordinary_reconnects")
                delay = self._backoff_delay(reconnect_attempt)
                reconnect_attempt += 1
                self._set_diagnostics(
                    twitch_connection_state="backoff",
                    twitch_reconnect_attempt=reconnect_attempt,
                    twitch_last_backoff_seconds=delay,
                )
                await self._async_sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._buffer.clear()
            self._set_diagnostics(
                twitch_connection_state="failed",
                twitch_last_error=_safe_exception_code(exc),
            )
        finally:
            self._set_network_available(False, reason="twitch_stopped")
            if connection is not None:
                await self._close_connection(connection)

    def _validate_token(self) -> None:
        validation = self._http.validate_token(self._credentials)
        self._increment("token_validations")
        if validation.client_id != self._credentials.client_id:
            raise TwitchHttpError(401, "token_client_id_mismatch")
        if "user:read:chat" not in set(validation.scopes):
            raise TwitchHttpError(403, "token_missing_user_read_chat")
        if validation.expires_in <= 0:
            raise TwitchHttpError(401, "access_token_expired")
        configured_user = str(self._credentials.eventsub_user_id or "")
        if configured_user and configured_user != validation.user_id:
            raise TwitchHttpError(401, "eventsub_user_id_mismatch")
        self._reader_user_id = configured_user or validation.user_id
        self._protocol.set_eventsub_user_id(self._reader_user_id)
        validation_delay = min(
            self._token_validation_interval_seconds,
            max(1.0, float(validation.expires_in) / 2.0),
        )
        self._next_token_validation = self._monotonic() + validation_delay
        self._set_diagnostics(twitch_token_expires_in=int(validation.expires_in))

    async def _connect_and_welcome(self, url: str) -> Tuple[EventSubWebSocket, EventSubWelcome]:
        connection: Optional[EventSubWebSocket] = None
        try:
            connection = await asyncio.wait_for(
                self._websocket_connector(url),
                timeout=self._open_timeout_seconds,
            )
            self._register_connection(connection)
            raw = await asyncio.wait_for(connection.recv(), timeout=self._welcome_timeout_seconds)
            result = self._protocol.parse(raw, received_monotonic=self._monotonic())
            if result.kind != "welcome" or result.welcome is None:
                if result.kind == "malformed":
                    self._increment("malformed_events")
                raise EventSubProtocolError("welcome_not_received")
            self._set_diagnostics(
                twitch_session_id=str(result.welcome.session_id)[:256],
                twitch_keepalive_timeout_seconds=float(result.welcome.keepalive_timeout_seconds),
            )
            return connection, result.welcome
        except BaseException:
            if connection is not None:
                await self._close_connection(connection)
            raise

    async def _consume_connection(
        self,
        connection: EventSubWebSocket,
        welcome: EventSubWelcome,
    ) -> _ConsumeOutcome:
        watchdog = EventSubSessionWatchdog(
            welcome.keepalive_timeout_seconds,
            observed_at=self._monotonic(),
        )
        receive_task: Optional[asyncio.Task[Any]] = asyncio.create_task(connection.recv())
        handoff_task: Optional[asyncio.Task[Tuple[EventSubWebSocket, EventSubWelcome]]] = None
        try:
            while not self._stop_requested.is_set():
                now = self._monotonic()
                if now >= self._next_token_validation:
                    try:
                        self._validate_token()
                    except TwitchHttpError as exc:
                        kind = "auth_failed" if exc.status in {401, 403} else "disconnect"
                        return _ConsumeOutcome(kind=kind, error_code=_safe_exception_code(exc))
                    now = self._monotonic()
                token_remaining = max(0.0, self._next_token_validation - now)
                timeout = max(0.01, min(watchdog.remaining(now), token_remaining or watchdog.remaining(now)))
                pending: set[asyncio.Task[Any]] = {receive_task}
                if handoff_task is not None:
                    pending.add(handoff_task)
                done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                now = self._monotonic()
                if not done:
                    if now >= self._next_token_validation:
                        try:
                            self._validate_token()
                        except TwitchHttpError as exc:
                            kind = "auth_failed" if exc.status in {401, 403} else "disconnect"
                            return _ConsumeOutcome(kind=kind, error_code=_safe_exception_code(exc))
                        continue
                    if watchdog.expired(now):
                        self._increment("keepalive_timeouts")
                        return _ConsumeOutcome(kind="disconnect", error_code="keepalive_timeout")
                    continue

                if handoff_task is not None and handoff_task in done:
                    try:
                        new_connection, new_welcome = handoff_task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        return _ConsumeOutcome(kind="disconnect", error_code=_safe_exception_code(exc))
                    if receive_task is not None and not receive_task.done():
                        receive_task.cancel()
                    if receive_task is not None:
                        await _await_cancelled(receive_task)
                    await self._close_connection(connection)
                    self._increment("reconnect_handoffs")
                    return _ConsumeOutcome(
                        kind="handoff",
                        connection=new_connection,
                        welcome=new_welcome,
                    )

                if receive_task not in done:
                    continue
                try:
                    raw = receive_task.result()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return _ConsumeOutcome(kind="disconnect", error_code=_safe_exception_code(exc))

                _, gate_epoch = self._buffer.gate_snapshot()
                result = self._protocol.parse(raw, received_monotonic=now)
                if result.kind != "malformed":
                    watchdog.observe(now)
                if result.kind in {
                    "notification",
                    "keepalive",
                    "duplicate",
                    "ignored",
                    "reconnect",
                    "revocation",
                }:
                    self._connection_activity_count += 1
                if result.kind == "notification" and result.message is not None:
                    self._increment("notifications_received")
                    if self._buffer.publish(result.message, gate_epoch=gate_epoch):
                        self._increment("notifications_enqueued")
                    self._set_diagnostics(
                        twitch_last_delivery_id=str(result.delivery_id)[:256],
                        twitch_last_event_id=str(result.event_id)[:256],
                    )
                elif result.kind == "keepalive":
                    self._increment("keepalives_received")
                elif result.kind == "duplicate":
                    self._increment("duplicate_events")
                elif result.kind == "malformed":
                    self._increment("malformed_events")
                    self._set_diagnostics(twitch_last_error=result.error_code)
                elif result.kind == "ignored":
                    self._increment("ignored_events")
                elif result.kind == "reconnect":
                    self._increment("reconnect_requests")
                    if handoff_task is None:
                        self._set_diagnostics(twitch_connection_state="reconnecting")
                        # The URL is intentionally passed through byte-for-byte
                        # and is never copied into diagnostics.
                        handoff_task = asyncio.create_task(self._connect_and_welcome(result.reconnect_url))
                elif result.kind == "revocation":
                    self._increment("revocations")
                    self._set_diagnostics(twitch_last_revocation_status=result.revocation_status)
                    auth_statuses = {
                        "authorization_revoked",
                        "chat_user_banned",
                        "moderator_removed",
                        "user_removed",
                    }
                    kind = "auth_failed" if result.revocation_status in auth_statuses else "revoked"
                    return _ConsumeOutcome(kind=kind, error_code=result.revocation_status)
                elif result.kind == "welcome":
                    self._increment("malformed_events")
                    self._set_diagnostics(twitch_last_error="unexpected_welcome")

                receive_task = asyncio.create_task(connection.recv())
            return _ConsumeOutcome(kind="disconnect", error_code="shutdown")
        finally:
            if receive_task is not None and not receive_task.done():
                receive_task.cancel()
                await _await_cancelled(receive_task)
            if handoff_task is not None and not handoff_task.done():
                handoff_task.cancel()
                await _await_cancelled(handoff_task)

    async def _default_websocket_connect(self, url: str) -> EventSubWebSocket:
        from websockets.asyncio.client import connect

        return await connect(
            url,
            open_timeout=self._open_timeout_seconds,
            close_timeout=2.0,
            ping_interval=None,
            max_size=self._max_frame_bytes,
        )

    def _backoff_delay(self, attempt: int) -> float:
        raw = min(self._backoff_max_seconds, self._backoff_base_seconds * (2 ** min(16, int(attempt))))
        jitter = 0.5 + max(0.0, min(1.0, float(self._random_value())))
        return min(self._backoff_max_seconds, raw * jitter)

    def _register_connection(self, connection: EventSubWebSocket) -> None:
        with self._connections_lock:
            if all(candidate is not connection for candidate in self._connections):
                self._connections.append(connection)

    async def _close_connection(self, connection: EventSubWebSocket) -> None:
        try:
            await asyncio.wait_for(connection.close(), timeout=2.0)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            with self._connections_lock:
                self._connections = [candidate for candidate in self._connections if candidate is not connection]

    async def _close_all_connections(self) -> None:
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            await self._close_connection(connection)

    def _set_network_available(
        self,
        available: bool,
        *,
        reason: str,
        force_epoch: bool = False,
    ) -> int:
        with self._phase_lock:
            changed = self._network_available != bool(available)
            self._network_available = bool(available)
            if not changed and not force_epoch:
                return self._buffer.clear() if not available else 0
            effective_accepting = self._phase_accepting and self._network_available
            return self._buffer.set_accepting(effective_accepting, reason=reason)

    def _increment(self, key: str, amount: int = 1) -> None:
        with self._diagnostics_lock:
            self._counts[str(key)] += int(amount)

    def _set_diagnostics(self, **values: Any) -> None:
        safe_values = {str(key): value for key, value in values.items()}
        with self._diagnostics_lock:
            self._diagnostics.update(safe_values)


async def _await_cancelled(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return
    except Exception:
        return


def _safe_exception_code(exc: BaseException) -> str:
    if isinstance(exc, TwitchHttpError):
        return _diagnostic_code(exc.code, "http_error")
    if isinstance(exc, EventSubProtocolError):
        return _diagnostic_code(exc.code, "eventsub_protocol_error")
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return f"websocket_close_{code}"
    name = type(exc).__name__.strip().lower()
    return _diagnostic_code(name, "network_error")
