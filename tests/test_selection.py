"""Hand-authored synthetic evidence; no semantic accuracy or fuzzing claim."""

from dataclasses import replace

import pytest

from memory_optimizer.domain import SourceMessage
from memory_optimizer.processing.selection import select_windows, validate_selection
from memory_optimizer.processing.types import Selection, SelectionConfig, SourceSpan


def test_windows_preserve_positions_roles_and_unknown_results_across_messages():
    messages = (
        SourceMessage("user", "背景🌱\r\n"),
        SourceMessage("user", "错误发生，还没修改", 0),
        SourceMessage("assistant", "已修改但未测试"),
        SourceMessage("assistant", "已修改但未测试"),
    )
    config = SelectionConfig(focus_chars=100, context_chars=4)
    result = select_windows(messages, config)
    validate_selection(result, messages, config)
    assert result.candidate_count == 3 and result.omitted_count == 0
    assert [w.focus for w in result.windows] == [
        SourceSpan(1, 0, 9),
        SourceSpan(2, 0, 7),
        SourceSpan(3, 0, 7),
    ]
    assert result.windows[0].reasons == ("change", "problem")
    assert result.windows[1].reasons == ("change", "verification")
    assert result.windows[0].context == (
        SourceSpan(0, 1, 5),
        SourceSpan(1, 0, 9),
        SourceSpan(2, 0, 4),
    )
    assert messages[1].timestamp == 0 and messages[2].timestamp is None
    assert select_windows(messages, config) == result


def test_boundary_spanning_signal_is_selected_and_context_contains_the_signal():
    messages = (SourceMessage("user", "🌱\r\n error followed by details"),)
    config = SelectionConfig(focus_chars=6, context_chars=6)
    result = select_windows(messages, config)
    validate_selection(result, messages, config)
    assert result.windows[0].focus == SourceSpan(0, 0, 6)
    context = result.windows[0].context[0]
    assert "error" in messages[0].content[context.start : context.end]


def test_limit_records_omitted_candidates_without_merging_distinct_occurrences():
    messages = tuple(SourceMessage("user", "error") for _ in range(7))
    config = SelectionConfig(max_windows=2, context_chars=0)
    result = select_windows(messages, config)
    validate_selection(result, messages, config)
    assert result.candidate_count == 7 and result.omitted_count == 5
    assert [w.focus.message_ordinal for w in result.windows] == [0, 1]
    assert all(w.context == (w.focus,) for w in result.windows)


def test_unselected_material_is_not_falsely_classified_as_failed_extraction():
    messages = (SourceMessage("user", "🌱hello world"),)
    result = select_windows(messages, SelectionConfig())
    assert result == Selection((), 0)


@pytest.mark.parametrize("mutation", ["outside", "unbounded", "duplicate", "coverage"])
def test_invalid_selector_cannot_publish_incorrect_evidence(mutation):
    messages = (SourceMessage("user", "error " + "x" * 200),)
    config = SelectionConfig(focus_chars=10, context_chars=2)
    result = select_windows(messages, config)
    window = result.windows[0]
    if mutation == "outside":
        result = replace(result, windows=(replace(window, focus=SourceSpan(1, 0, 5)),))
    elif mutation == "unbounded":
        result = replace(result, windows=(replace(window, context=(SourceSpan(0, 0, 200),)),))
    elif mutation == "duplicate":
        result = Selection((window, replace(window, ordinal=1)), 2)
    else:
        result = replace(result, candidate_count=2)
    with pytest.raises(ValueError, match="selection"):
        validate_selection(result, messages, config)


@pytest.mark.parametrize(
    "limits",
    [
        {"focus_chars": 0},
        {"context_chars": -1},
        {"max_windows": 0},
        {"max_windows": 257},
        {"focus_chars": True},
        {"context_chars": 2001},
    ],
)
def test_invalid_selection_limits_are_rejected(limits):
    with pytest.raises(ValueError, match="limit"):
        SelectionConfig(**limits)
