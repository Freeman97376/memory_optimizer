"""Deterministic evidence selection; matches are relevance signals, never facts."""

import re
from itertools import groupby

from memory_optimizer.domain import SourceMessage
from memory_optimizer.processing.types import Selection, SelectionConfig, SourceSpan, Window

# Change the version when changing matching or context semantics; retain old versions
# while any unfinished run uses them. Never use a model to execute this source material.
SIGNALS = re.compile(
    r"(?P<problem>\b(?:error|exception|traceback|bug|issue|fail(?:ed|ure)?|"
    r"timeout|broken|crash(?:ed)?)\b|错误|异常|失败|报错|超时|问题)"
    r"|(?P<change>\b(?:fix(?:ed)?|patch(?:ed)?|chang(?:e|ed)|revert(?:ed)?|"
    r"rollback|refactor(?:ed)?)\b|修改|修复|改动|回滚|重构)"
    r"|(?P<verification>\b(?:test(?:s|ed|ing)?|pytest|assert(?:ion)?|"
    r"verif(?:y|ied|ication)|pass(?:ed)?)\b|测试|验证|通过)",
    re.IGNORECASE,
)


def select_windows(messages: tuple[SourceMessage, ...], config: SelectionConfig) -> Selection:
    windows = []
    candidate_count = 0
    for ordinal, message in enumerate(messages):
        # Stream matches and groups; the number of retained objects is capped even
        # when the entire input is full of signals. Repeated text is not deduplicated.
        groups = groupby(
            SIGNALS.finditer(message.content),
            key=lambda match: match.start() // config.focus_chars,
        )
        for block, matches in groups:
            reasons = tuple(sorted({match.lastgroup for match in matches}))
            candidate_count += 1
            if len(windows) == config.max_windows:
                continue
            start = block * config.focus_chars
            end = min(start + config.focus_chars, len(message.content))
            focus = SourceSpan(ordinal, start, end)
            radius = config.context_chars
            context = []
            if start == 0 and ordinal > 0 and radius:
                previous = messages[ordinal - 1].content
                context.append(
                    SourceSpan(ordinal - 1, max(0, len(previous) - radius), len(previous))
                )
            context.append(
                SourceSpan(ordinal, max(0, start - radius), min(len(message.content), end + radius))
            )
            if end == len(message.content) and ordinal + 1 < len(messages) and radius:
                context.append(
                    SourceSpan(ordinal + 1, 0, min(radius, len(messages[ordinal + 1].content)))
                )
            windows.append(Window(len(windows), focus, tuple(context), reasons))
    return Selection(tuple(windows), candidate_count)


def validate_selection(
    selection: Selection, messages: tuple[SourceMessage, ...], config: SelectionConfig
) -> None:
    """Keep invalid replacement strategies from publishing dangling source references."""
    if (
        type(selection.candidate_count) is not int
        or selection.candidate_count < len(selection.windows)
        or len(selection.windows) > config.max_windows
        or len(selection.windows) != min(selection.candidate_count, config.max_windows)
    ):
        raise ValueError("invalid selection coverage")
    seen = set()
    for ordinal, window in enumerate(selection.windows):
        if (
            window.ordinal != ordinal
            or window.focus in seen
            or not window.reasons
            or not set(window.reasons) <= {"problem", "change", "verification"}
            or not 1 <= len(window.context) <= 3
        ):
            raise ValueError("invalid selection window")
        seen.add(window.focus)
        for span in (window.focus, *window.context):
            if not (
                type(span.message_ordinal) is int
                and type(span.start) is int
                and type(span.end) is int
                and 0 <= span.message_ordinal < len(messages)
                and 0 <= span.start < span.end <= len(messages[span.message_ordinal].content)
            ):
                raise ValueError("invalid selection source reference")
        if window.focus.end - window.focus.start > config.focus_chars or not any(
            span.message_ordinal == window.focus.message_ordinal
            and span.start <= window.focus.start < window.focus.end <= span.end
            for span in window.context
        ):
            raise ValueError("invalid selection context")
        if len(set(window.context)) != len(window.context):
            raise ValueError("duplicate selection context")
        for span in window.context:
            if span.message_ordinal == window.focus.message_ordinal:
                bounded = (
                    span.start >= max(0, window.focus.start - config.context_chars)
                    and span.end <= window.focus.end + config.context_chars
                )
            else:
                bounded = (
                    abs(span.message_ordinal - window.focus.message_ordinal) == 1
                    and span.end - span.start <= config.context_chars
                )
            if not bounded:
                raise ValueError("unbounded selection context")
