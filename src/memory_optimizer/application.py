"""Application workflows own strategy; repositories own atomic persistence."""

from dataclasses import dataclass

from memory_optimizer.domain import (
    IngestCommand,
    IngestPlan,
    IngestReceipt,
    SearchCandidate,
    SearchQuery,
    SourceSlice,
    payload_digest,
)
from memory_optimizer.ports import (
    IngestRepository,
    SearchIndex,
    SourceSlicer,
    StorageLifecycle,
)
from memory_optimizer.processing.ports import AddProcessing


def source_slices(content: str, slicer: SourceSlicer) -> tuple[SourceSlice, ...]:
    """A replacement strategy must preserve source coverage and exact offsets."""
    parts = []
    covered = 0
    for ordinal, part in enumerate(slicer(content)):
        if not (
            part.ordinal == ordinal
            and 0 <= part.start <= covered < part.end <= len(content)
            and part.content == content[part.start : part.end]
        ):
            raise ValueError("source slicer returned an invalid or non-progressing slice")
        parts.append(part)
        covered = part.end
    if covered != len(content):
        raise ValueError("source slicer did not cover the complete message")
    return tuple(parts)


@dataclass(frozen=True)
class AddService:
    repository: IngestRepository
    slicer: SourceSlicer
    processing: AddProcessing | None = None

    def add(self, command: IngestCommand) -> IngestReceipt:
        plan = IngestPlan(
            command=command,
            payload_hash=payload_digest(command),
            message_slices=tuple(
                source_slices(message.content, self.slicer) for message in command.messages
            ),
            processing=self.processing.spec if self.processing else None,
        )
        receipt = self.repository.commit(plan)
        if self.processing:
            self.processing.finish(command, receipt)
        return receipt


@dataclass(frozen=True)
class SearchService:
    index: SearchIndex

    def search(self, query: SearchQuery) -> list[SearchCandidate]:
        return self.index.search(query)


@dataclass(frozen=True)
class Services:
    add: AddService
    search: SearchService
    lifecycle: StorageLifecycle
