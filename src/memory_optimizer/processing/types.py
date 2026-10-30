"""Immutable processing contracts. Source positions count Unicode code points."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SelectionConfig:
    focus_chars: int = 2000
    context_chars: int = 300
    max_windows: int = 32

    def __post_init__(self):
        for value, minimum, maximum in (
            (self.focus_chars, 1, 16_000),
            (self.context_chars, 0, 2000),
            (self.max_windows, 1, 256),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("invalid selection limit")


@dataclass(frozen=True)
class ProcessingSpec:
    flow_version: str = "selection-v1"
    selector_version: str = "coding-signals-v1"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    processor_version: str | None = None

    def __post_init__(self):
        if any(
            type(value) is not str or not 1 <= len(value) <= 100
            for value in (self.flow_version, self.selector_version)
        ):
            raise ValueError("invalid processing version")
        if not isinstance(self.selection, SelectionConfig):
            raise ValueError("invalid selection configuration")
        if self.processor_version is not None and (
            type(self.processor_version) is not str or not 1 <= len(self.processor_version) <= 100
        ):
            raise ValueError("invalid processor version")


@dataclass(frozen=True)
class SourceSpan:
    message_ordinal: int
    start: int
    end: int


@dataclass(frozen=True)
class Window:
    ordinal: int
    focus: SourceSpan
    context: tuple[SourceSpan, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    windows: tuple[Window, ...]
    candidate_count: int

    @property
    def omitted_count(self) -> int:
        return self.candidate_count - len(self.windows)


@dataclass(frozen=True)
class RunLease:
    run_id: str
    batch_id: str
    user_id: str
    owner: str | None
    fence: int
    stage: Literal["raw_ready", "selected", "complete"]
    spec: ProcessingSpec | None
