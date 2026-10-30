"""Closed selector/validator pairing: legacy v1 or full deterministic source plans."""

from collections.abc import Callable
from dataclasses import dataclass

from memory_optimizer.domain import ProcessingUnavailable, SourceMessage
from memory_optimizer.processing.plan_types import (
    PLAN_SCHEMA,
    PlanningSpec,
    SelectionPlan,
    require_planning_spec,
)
from memory_optimizer.processing.plan_validation import (
    validate_final_contract as validate_final_contract,
)
from memory_optimizer.processing.plan_validation import validate_plan
from memory_optimizer.processing.plan_validation import (
    validate_plan_contract as validate_plan_contract,
)
from memory_optimizer.processing.selection import validate_selection
from memory_optimizer.processing.types import ProcessingSpec, Selection, SelectionConfig


def selection_contract(spec: ProcessingSpec | PlanningSpec) -> str:
    """Resolve an exact supported combination, never fall back to the v1 validator."""
    if type(spec) is ProcessingSpec:
        if (
            spec.flow_version in ("selection-v1", "offline-candidates-v1")
            and spec.selector_version == "coding-signals-v1"
            and type(spec.selection) is SelectionConfig
        ):
            return "legacy-selection-v1"
    elif type(spec) is PlanningSpec:
        require_planning_spec(spec)
        return PLAN_SCHEMA
    raise ProcessingUnavailable("unsupported selection contract combination")


@dataclass(frozen=True)
class SelectionBinding:
    spec: ProcessingSpec | PlanningSpec
    selector: Callable

    def __post_init__(self):
        selection_contract(self.spec)
        if not callable(self.selector):
            raise ValueError("selector must be callable")

    def select(self, messages: tuple[SourceMessage, ...]) -> Selection | SelectionPlan:
        contract = selection_contract(self.spec)
        if contract == "legacy-selection-v1":
            result = self.selector(messages, self.spec.selection)
            # Guard before invoking v1's field/coverage rules on another result family.
            if type(result) is not Selection:
                raise ValueError("selector result does not match legacy selection contract")
            validate_selection(result, messages, self.spec.selection)
        else:
            result = self.selector(messages, self.spec)
            validate_plan(result, messages, self.spec)
        return result
