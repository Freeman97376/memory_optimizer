"""Resume durable stages. No HTTP, SQL, model SDK, or background worker dependency."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from time import monotonic, sleep

from memory_optimizer.domain import (
    IngestCommand,
    IngestReceipt,
    ProcessingBusy,
    ProcessingUnavailable,
)
from memory_optimizer.processing.ports import (
    ProcessingRepository,
    SelectedProcessor,
    WindowSelector,
)
from memory_optimizer.processing.selection import select_windows
from memory_optimizer.processing.selection_contracts import SelectionBinding
from memory_optimizer.processing.types import ProcessingSpec


@dataclass(frozen=True)
class ProcessingFlow:
    repository: ProcessingRepository
    spec: ProcessingSpec = field(default_factory=ProcessingSpec)
    selectors: Mapping[str, WindowSelector] = field(
        default_factory=lambda: {"coding-signals-v1": select_windows}
    )
    lease_seconds: int = 30
    wait_seconds: float = 1.0
    processors: Mapping[str, SelectedProcessor] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.spec) is not ProcessingSpec:
            raise ProcessingUnavailable("planning requires a separate processing flow")
        if not 1 <= self.lease_seconds <= 300 or not 0 <= self.wait_seconds <= 5:
            raise ValueError("invalid processing coordination limits")

    def finish(self, command: IngestCommand, receipt: IngestReceipt) -> None:
        deadline = monotonic() + self.wait_seconds
        while True:
            try:
                lease = self.repository.claim(command.user_id, receipt.batch_id, self.lease_seconds)
                break
            except ProcessingBusy:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise
                sleep(min(0.05, remaining))
        if lease.stage == "complete":
            return
        spec = lease.spec
        # PlanningSpec needs its own durable stages; it cannot enter this legacy flow.
        if type(spec) is not ProcessingSpec:
            raise ProcessingUnavailable("unsupported unfinished processing version")
        processor = self.processors.get(spec.flow_version)
        if spec.flow_version != "selection-v1" and processor is None:
            raise ProcessingUnavailable("unsupported unfinished processing version")
        if processor is not None:
            processor.check_version(spec)
        if lease.stage == "raw_ready":
            selector = self.selectors.get(spec.selector_version)
            if selector is None:
                raise ProcessingUnavailable("unsupported unfinished selector version")
            selected = SelectionBinding(spec, selector).select(command.messages)
            self.repository.checkpoint(lease, selected)
            lease = replace(lease, stage="selected")
        if processor is None:
            self.repository.finish_raw(lease)
        else:
            processor.process(lease, command)
