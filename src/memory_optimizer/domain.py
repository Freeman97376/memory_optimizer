"""Internal values and errors, independent of HTTP, database drivers and model SDKs."""

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from memory_optimizer.processing.coverage_types import CoverageSpec
    from memory_optimizer.processing.types import ProcessingSpec


@dataclass(frozen=True)
class SourceMessage:
    role: Literal["user", "assistant"]
    content: str
    timestamp: int | None = None


@dataclass(frozen=True)
class IngestCommand:
    request_id: str
    user_id: str
    session_id: str
    messages: tuple[SourceMessage, ...]


@dataclass(frozen=True)
class SourceSlice:
    ordinal: int
    start: int
    end: int
    content: str


@dataclass(frozen=True)
class IngestPlan:
    command: IngestCommand
    payload_hash: str
    message_slices: tuple[tuple[SourceSlice, ...], ...]
    processing: "ProcessingSpec | CoverageSpec | None" = None


@dataclass(frozen=True)
class IngestReceipt:
    request_id: str
    user_id: str
    session_id: str
    batch_id: str
    reused: bool


@dataclass(frozen=True)
class SearchQuery:
    query: str
    user_id: str
    top_k: int = 100
    options: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SearchCandidate:
    id: str
    content: str
    score: float


class RequestConflict(Exception):
    """The scope/request key already belongs to a different payload."""


class StorageUnavailable(Exception):
    """A storage operation or readiness check could not complete."""


class ProcessingUnavailable(Exception):
    """Processing cannot safely complete; a raw commit is not a terminal receipt."""


class ProcessingBusy(ProcessingUnavailable):
    """Another live lease owns the run, or this owner's lease is no longer valid."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_digest(command: IngestCommand) -> str:
    """Preserve P1's canonical JSON exactly, including omitted/null timestamps."""
    messages = []
    for message in command.messages:
        value = {"role": message.role, "content": message.content}
        if message.timestamp is not None:
            value["timestamp"] = message.timestamp
        messages.append(value)
    return digest(
        json.dumps(
            {
                "request_id": command.request_id,
                "user_id": command.user_id,
                "session_id": command.session_id,
                "messages": messages,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
