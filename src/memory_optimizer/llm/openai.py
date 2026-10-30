"""Official text-only API adapter. No retries, redirects, tools or request logging."""

import asyncio
import json

import httpx
from pydantic import SecretStr

from memory_optimizer.llm.types import MODEL, Result


class OpenAITransport:
    def __init__(self, api_key: SecretStr, *, transport=None):
        if not api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required")
        self._key = api_key
        self._transport = transport

    async def complete(self, body, timeout):
        try:
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(
                    transport=self._transport,
                    follow_redirects=False,
                    trust_env=False,
                    timeout=httpx.Timeout(timeout, connect=min(timeout, 10)),
                    headers={"Authorization": "Bearer " + self._key.get_secret_value()},
                ) as client:
                    async with client.stream(
                        "POST", "https://api.openai.com/v1/chat/completions", json=body
                    ) as response:
                        if response.status_code != 200:
                            return Result("error", error="http_" + str(response.status_code))
                        chunks, size = [], 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > 1048576:
                                return Result("error", error="response_too_large")
                            chunks.append(chunk)
                        value = json.loads(b"".join(chunks))
            return decode(value, body["max_completion_tokens"])
        except (TimeoutError, httpx.HTTPError, ValueError, RecursionError):
            return Result("error", error="transport_unknown")


def decode(value, max_output):
    if type(value) is not dict or value.get("model") != MODEL:
        return Result("error", error="unexpected_model")
    usage = value.get("usage")
    usage = usage if type(usage) is dict else {}
    inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not (
        type(inp) is int and 0 <= inp <= 128000 and type(out) is int and 0 <= out <= max_output
    ):
        inp = out = None
    choices = value.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        return Result("error", input_tokens=inp, output_tokens=out, error="invalid_response")
    choice = choices[0]
    message = choice.get("message")
    if type(message) is not dict:
        return Result("error", input_tokens=inp, output_tokens=out, error="invalid_response")
    status = "complete"
    content = message.get("content")
    if message.get("refusal") or choice.get("finish_reason") == "content_filter":
        status, content = "refused", None
    elif choice.get("finish_reason") == "length":
        status, content = "truncated", None
    elif (
        choice.get("finish_reason") != "stop"
        or type(content) is not str
        or not content
        or len(content.encode("utf-8")) > 65536
    ):
        status, content = "error", None
    return Result(status, content, inp, out)
