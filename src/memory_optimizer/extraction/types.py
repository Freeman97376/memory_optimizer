from dataclasses import asdict, dataclass
from typing import Protocol

from memory_optimizer.domain import SourceMessage, digest
from memory_optimizer.processing.types import RunLease, SourceSpan, Window

SCHEMA_VERSION = "episode-candidates-v1"
MAX_RESPONSE_BYTES = 64 * 1024
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class EvidenceText:
    span: SourceSpan
    content: str
    role: str
    source_timestamp: int | None


@dataclass(frozen=True)
class ExtractionInput:
    user_id: str
    batch_id: str
    run_id: str
    window_id: str
    window: Window
    payload_hash: str
    processor_version: str
    configuration_hash: str
    sources: tuple[EvidenceText, ...]
    schema_version: str = SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        from memory_optimizer.extraction.json_codec import canonical

        return digest(canonical(asdict(self)))


def prepare_input(
    lease: RunLease,
    window_id: str,
    window: Window,
    messages: tuple[SourceMessage, ...],
    payload_hash: str,
) -> ExtractionInput:
    from memory_optimizer.extraction.json_codec import canonical

    sources = tuple(
        EvidenceText(
            span,
            messages[span.message_ordinal].content[span.start : span.end],
            messages[span.message_ordinal].role,
            messages[span.message_ordinal].timestamp,
        )
        for span in window.context
    )
    return ExtractionInput(
        lease.user_id,
        lease.batch_id,
        lease.run_id,
        window_id,
        window,
        payload_hash,
        lease.spec.processor_version,
        digest(canonical(asdict(lease.spec))),
        sources,
    )


@dataclass(frozen=True)
class Citation:
    span: SourceSpan
    quote: str
    role: str
    source_timestamp: int | None


@dataclass(frozen=True)
class Claim:
    text: str
    evidence: tuple[Citation, ...]
    attribution: str


@dataclass(frozen=True)
class ObjectMention:
    name: Claim
    kind: str
    qualifiers: tuple[Claim, ...]


@dataclass(frozen=True)
class Observation:
    content: Claim
    kind: str
    event_time: Claim | None
    corrects_attempt: int | None


@dataclass(frozen=True)
class Attempt:
    action: Claim
    kind: str
    event_time: Claim | None
    before: Claim | None
    after: Claim | None
    object_refs: tuple[int, ...]
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class EpisodeCandidate:
    problem: Claim | None
    conditions: tuple[Claim, ...]
    objects: tuple[ObjectMention, ...]
    attempts: tuple[Attempt, ...]


@dataclass(frozen=True)
class ValidatedCandidate:
    candidate: EpisodeCandidate
    fingerprint: str
    evidence_key: str


@dataclass(frozen=True)
class ExtractionReply:
    status: str
    content: str | None = None


class OfflineExtractor(Protocol):
    version: str

    def extract(self, request: ExtractionInput, attempt: int) -> ExtractionReply: ...


class CandidateValidationError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
