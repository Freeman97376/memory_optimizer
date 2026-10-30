"""Authenticate first and bound input before JSON validation."""

import asyncio
import hmac
import ipaddress
import json

from starlette.responses import JSONResponse

from memory_optimizer.config import Settings


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError("non-finite JSON number")


class RequestGuard:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    def authenticated(self, scope) -> bool:
        expected = self.settings.api_key
        if expected is None:
            client = scope.get("client")
            try:
                return (
                    self.settings.allow_local_anonymous
                    and bool(client)
                    and (ipaddress.ip_address(client[0]).is_loopback)
                )
            except ValueError:
                return False
        provided = []
        for name, value in scope["headers"]:
            if name == b"authorization":
                parts = value.split()
                if len(parts) != 2 or parts[0].lower() not in {b"bearer", b"token"}:
                    return False
                provided.append(parts[1])
            elif name == b"x-api-key":
                provided.append(value)
        return len(provided) == 1 and hmac.compare_digest(
            provided[0], expected.get_secret_value().encode("utf-8")
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") not in {"/add", "/search"}:
            return await self.app(scope, receive, send)

        async def reject(status, code):
            await JSONResponse({"error": code}, status_code=status)(scope, receive, send)

        if not self.authenticated(scope):
            return await reject(401, "unauthorized")
        headers = dict(scope["headers"])
        if headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower() != (
            b"application/json"
        ):
            return await reject(415, "application_json_required")
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0:
                raise ValueError()
        except ValueError:
            return await reject(400, "invalid_content_length")
        if length > self.settings.max_body_bytes:
            return await reject(413, "request_too_large")
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    piece = event.get("body", b"")
                    if len(body) + len(piece) > self.settings.max_body_bytes:
                        return await reject(413, "request_too_large")
                    body.extend(piece)
                    if not event.get("more_body", False):
                        break
        except TimeoutError:
            return await reject(408, "request_body_timeout")
        try:
            json.loads(
                body.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=invalid_constant,
            )
        except (ValueError, RecursionError):
            return await reject(400, "invalid_json")
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
