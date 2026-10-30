"""One reservation per outbound call; no hidden retry or transaction over I/O."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

from memory_optimizer.domain import digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.llm.types import MODEL, CallBlocked, Limits, Result


class BudgetedModel:
    def __init__(self, ledger, transport, limits=None, *, clock=None):
        self.ledger, self.transport = ledger, transport
        self.limits = limits or Limits()
        self.clock = clock or (lambda: datetime.now(UTC))

    def call(self, *, scope, phase, messages, schema, deadline, validate, max_output=8192):
        if not scope or phase not in {"extract", "relation"}:
            raise ValueError("invalid provider call scope")
        if type(max_output) is not int or not 1 <= max_output <= 8192:
            raise ValueError("invalid output limit")
        if deadline.tzinfo is None or deadline <= self.clock():
            raise CallBlocked("deadline_exhausted")
        if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
            raise ValueError("provider messages must be fixed system plus source data")
        if any(type(m.get("content")) is not str for m in messages):
            raise ValueError("only text provider input is supported")
        body = {
            "model": MODEL,
            "messages": messages,
            "n": 1,
            "temperature": 0,
            "stream": False,
            "store": False,
            "service_tier": "default",
            "max_completion_tokens": max_output,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "memory_" + phase, "strict": True, "schema": schema},
            },
        }
        encoded = canonical(body)
        if len(encoded.encode("utf-8")) > 300000:
            raise CallBlocked("provider_input_too_large")
        # The cache binds scope, actual source/context, schema, prompt, model and settings.
        key = digest(phase + "\n" + encoded)
        reserved = self.ledger.reserve(self.limits, scope, key, deadline, max_output)
        if isinstance(reserved, Result):
            validate(reserved.content)
            return reserved
        seconds = min(90.0, (reserved.expires_at - self.clock()).total_seconds())
        if seconds <= 0:
            result = Result("error", error="deadline_exhausted")
        else:
            try:
                result = asyncio.run(self.transport.complete(body, seconds))
            except (TimeoutError, OSError):
                result = Result("error", error="transport_unknown")
        if result.status == "complete":
            try:
                validate(result.content)
            except ValueError as exc:
                from memory_optimizer.extraction.types import CandidateValidationError

                code = exc.code if isinstance(exc, CandidateValidationError) else "invalid_evidence"
                result = replace(result, status="error", content=None, error=code)
        if self.clock() >= reserved.expires_at:
            result = replace(result, status="error", content=None, error="late_reply")
        # Persist accounting before returning any answer; commit failures propagate.
        return self.ledger.finish(reserved, result)
