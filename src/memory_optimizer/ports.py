"""Small synchronous boundaries; implementations may share one connection pool."""

from collections.abc import Callable, Iterable
from typing import Protocol

from memory_optimizer.domain import (
    IngestPlan,
    IngestReceipt,
    SearchCandidate,
    SearchQuery,
    SourceSlice,
)

SourceSlicer = Callable[[str], Iterable[SourceSlice]]


class IngestRepository(Protocol):
    def commit(self, plan: IngestPlan) -> IngestReceipt:
        """Atomically commit sources/slices, replay an equal request, or raise a conflict."""
        ...


class SearchIndex(Protocol):
    def search(self, query: SearchQuery) -> list[SearchCandidate]: ...


class StorageLifecycle(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def check_ready(self) -> None: ...
