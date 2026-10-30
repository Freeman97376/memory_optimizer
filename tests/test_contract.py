import asyncio
import json

import pytest
from pydantic import SecretStr, ValidationError

from memory_optimizer.api.guard import RequestGuard
from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.chunking import chunks
from memory_optimizer.config import Settings
from memory_optimizer.domain import payload_digest
from memory_optimizer.models import AddRequest, SearchRequest


def payload():
    return {
        "request_id": " r1 ",
        "user_id": "u",
        "session_id": "s",
        "messages": [{"role": "user", "content": "  def cache():\r\n    return '中文🌱'\n"}],
    }


@pytest.mark.parametrize("field", ["request_id", "user_id", "session_id", "messages"])
def test_add_requires_fields(field):
    value = payload()
    del value[field]
    with pytest.raises(ValidationError):
        AddRequest.model_validate(value)


@pytest.mark.parametrize(
    "update",
    [
        {"messages": []},
        {"request_id": ""},
        {"user_id": 1},
        {"metadata": {}},
        {"messages": [{"role": "system", "content": "text"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": {"text": "text"}}]},
        {"messages": [{"role": "user", "content": "\x00"}]},
        {"messages": [{"role": "user", "content": "\ud800"}]},
    ],
)
def test_invalid_input_is_explicit(update):
    with pytest.raises(ValidationError):
        AddRequest.model_validate(payload() | update)


@pytest.mark.parametrize("value", [True, 1.5, "123", 2**63, -(2**63) - 1])
def test_timestamp_does_not_coerce(value):
    data = payload()
    data["messages"][0]["timestamp"] = value
    with pytest.raises(ValidationError):
        AddRequest.model_validate(data)


def test_raw_text_and_opaque_ids_preserved():
    data = payload()
    parsed = AddRequest.model_validate(data)
    assert parsed.request_id == " r1 "
    assert parsed.messages[0].content.encode() == data["messages"][0]["content"].encode()
    with_null = payload()
    with_null["messages"][0]["timestamp"] = None
    assert payload_digest(ingest_command(parsed)) == payload_digest(
        ingest_command(AddRequest.model_validate(with_null))
    )
    data["messages"][0]["content"] += " "
    assert payload_digest(ingest_command(parsed)) != payload_digest(
        ingest_command(AddRequest.model_validate(data))
    )


@pytest.mark.parametrize("length", [0, 1, 19, 20, 21, 37, 40, 111])
def test_chunks_reconstruct_original(length):
    content = ("🌱中文\r\n  def cache(): e\u0301\n" * 10)[:length]
    parts = list(chunks(content, 20, 3))
    recovered = ""
    covered = 0
    for part in parts:
        assert part.content == content[part.start : part.end]
        assert 0 < len(part.content) <= 20
        assert part.start <= covered
        recovered += part.content[max(0, covered - part.start) :]
        covered = part.end
    assert recovered.encode() == content.encode()


@pytest.mark.parametrize("top_k", [0, -1, 101, True, "100"])
def test_search_limit_strict(top_k):
    with pytest.raises(ValidationError):
        SearchRequest(query="cache", user_id="u", top_k=top_k)


def call_guard(headers, pieces, max_body=1000, client=("127.0.0.1", 1)):
    output = []
    called = []
    settings = Settings(
        database_url=SecretStr("postgresql://unused"),
        api_key=SecretStr("local-test-key"),
        max_body_bytes=max_body,
    )

    async def app(scope, receive, send):
        called.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})

    events = iter(
        [
            {"type": "http.request", "body": part, "more_body": i < len(pieces) - 1}
            for i, part in enumerate(pieces)
        ]
    )

    async def receive():
        return next(events)

    async def send(event):
        output.append(event)

    asyncio.run(
        RequestGuard(app, settings)(
            {"type": "http", "path": "/add", "headers": headers, "client": client}, receive, send
        )
    )
    return output[0]["status"], called


JSON_HEADERS = [(b"content-type", b"application/json")]
AUTH_HEADERS = JSON_HEADERS + [(b"authorization", b"Bearer local-test-key")]


@pytest.mark.parametrize(
    "headers,status",
    [
        (JSON_HEADERS, 401),
        (JSON_HEADERS + [(b"authorization", b"Bearer wrong")], 401),
        (AUTH_HEADERS + [(b"x-api-key", b"local-test-key")], 401),
        ([(b"authorization", b"Bearer local-test-key")], 415),
        (AUTH_HEADERS + [(b"content-length", b"999999")], 413),
    ],
)
def test_guard_rejects_before_reading_body(headers, status):
    actual, called = call_guard(headers, [])
    assert actual == status
    assert not called


@pytest.mark.parametrize(
    "data",
    [
        b'{"id":1,"id":2}',
        b'{"nested":{"id":1,"id":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b"\xff",
        b"not-json",
    ],
)
def test_invalid_json_rejected_without_echo(data):
    status, called = call_guard(AUTH_HEADERS, [data])
    assert status == 400
    assert not called


def test_stream_limit_and_exact_boundary():
    body = json.dumps(payload()).encode()
    status, called = call_guard(AUTH_HEADERS, [body[:20], body[20:]], len(body) - 1)
    assert status == 413 and not called
    status, called = call_guard(AUTH_HEADERS, [body[:20], body[20:]], len(body))
    assert status == 204 and called[0]["body"] == body


def test_service_requires_explicit_auth_mode():
    with pytest.raises(ValueError, match="MEMORY_API_KEY"):
        Settings(database_url=SecretStr("postgresql://unused")).validate_service()


def test_local_anonymous_rejects_nonloopback_and_forwarded_headers():
    guard = RequestGuard(
        None, Settings(database_url=SecretStr("postgresql://unused"), allow_local_anonymous=True)
    )
    assert guard.authenticated({"client": ("127.0.0.1", 1), "headers": []})
    assert not guard.authenticated(
        {"client": ("203.0.113.9", 1), "headers": [(b"x-forwarded-for", b"127.0.0.1")]}
    )
