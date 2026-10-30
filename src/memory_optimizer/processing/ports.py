from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from memory_optimizer.processing.coverage_types import CoverageSpec

from memory_optimizer.domain import IngestCommand, IngestReceipt, SourceMessage
from memory_optimizer.processing.types import ProcessingSpec, RunLease, Selection, SelectionConfig

WindowSelector = Callable[[tuple[SourceMessage, ...], SelectionConfig], Selection]


class AddProcessing(Protocol):
    spec: "ProcessingSpec | CoverageSpec"

    def finish(self, command: IngestCommand, receipt: IngestReceipt) -> None: ...


class SelectedProcessor(Protocol):
    def check_version(self, spec: ProcessingSpec) -> None: ...

    def process(self, lease: RunLease, command: IngestCommand) -> None: ...


class ProcessingRepository(Protocol):
    def claim(self, user_id: str, batch_id: str, lease_seconds: int) -> RunLease:
        """Return a terminal run, claim an available run, or raise ProcessingBusy."""
        ...

    def checkpoint(self, lease: RunLease, selection: Selection) -> None:
        """Atomically save all selected windows and advance raw_ready to selected."""
        ...

    def finish_raw(self, lease: RunLease) -> None:
        """Atomically defer selected windows and record the fixed raw-only terminal."""
        ...
