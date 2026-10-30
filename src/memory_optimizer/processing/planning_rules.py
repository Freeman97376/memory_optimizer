"""Frozen v2 source, segmentation and signal rules; no I/O or execution clock."""

import hashlib
import json
import re
from itertools import groupby

from memory_optimizer.domain import SourceMessage
from memory_optimizer.processing.plan_types import PlanLimits
from memory_optimizer.processing.types import SourceSpan

# Independent of coding-signals-v1: changing either strategy must not mutate the other.
_SIGNALS = re.compile(
    r"(?P<problem>\b(?:error|exception|traceback|bug|issue|fail(?:ed|ure)?|"
    r"timeout|broken|crash(?:ed)?)\b|错误|异常|失败|报错|超时|问题)"
    r"|(?P<change>\b(?:fix(?:ed)?|patch(?:ed)?|chang(?:e|ed)|revert(?:ed)?|"
    r"rollback|refactor(?:ed)?)\b|修改|修复|改动|回滚|重构)"
    r"|(?P<verification>\b(?:test(?:s|ed|ing)?|pytest|assert(?:ion)?|"
    r"verif(?:y|ied|ication)|pass(?:ed)?)\b|测试|验证|通过)",
    re.IGNORECASE,
)


def source_digest(messages: tuple[SourceMessage, ...]) -> str:
    """Hash every source field/order, streaming per message; not a scope/cache key.

    Hashing and source accounting read the full input. max_scan_chars bounds rule
    classification only, not input hashing, validation CPU or future model tokens.
    """
    if type(messages) is not tuple:
        raise ValueError("planning requires immutable source messages")
    result = hashlib.sha256(b"planning-source-v1\n[")
    for ordinal, message in enumerate(messages):
        if (
            type(message) is not SourceMessage
            or type(message.role) is not str
            or message.role not in ("user", "assistant")
            or type(message.content) is not str
            or (message.timestamp is not None and type(message.timestamp) is not int)
        ):
            raise ValueError("invalid planning source message")
        if ordinal:
            result.update(b",")
        result.update(
            json.dumps(
                {"content": message.content, "role": message.role, "timestamp": message.timestamp},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    result.update(b"]")
    return result.hexdigest()


def inspection_ends(messages: tuple[SourceMessage, ...], limits: PlanLimits):
    """Grant a global source prefix of whole focus units, never skip a large unit."""
    remaining = limits.max_scan_chars
    stopped = False
    for message in messages:
        size = len(message.content)
        if stopped:
            end = 0
        elif size <= remaining:
            end = size
        else:
            end = (remaining // limits.focus_chars) * limits.focus_chars
            stopped = True
        remaining -= end
        yield end


def signal_blocks(content: str, end: int, focus_chars: int):
    """Stream complete matches by start block inside the inspected prefix.

    At a truncated prefix, reject matches touching its artificial right edge:
    no look-ahead into uninspected text and no false word boundary at the cut.
    A match crossing two inspected units belongs to its start unit, as in v1.
    """
    matches = (
        match
        for match in _SIGNALS.finditer(content[:end])
        if end == len(content) or match.end() < end
    )
    for block, group in groupby(matches, key=lambda match: match.start() // focus_chars):
        yield block, tuple(sorted({match.lastgroup for match in group}))


def window_context(
    messages: tuple[SourceMessage, ...], focus: SourceSpan, radius: int
) -> tuple[SourceSpan, ...]:
    """Exact same-message radius plus adjacent tails/heads at message edges."""
    ordinal = focus.message_ordinal
    size = len(messages[ordinal].content)
    spans = []
    if focus.start == 0 and ordinal > 0 and radius:
        previous = len(messages[ordinal - 1].content)
        if previous:
            spans.append(SourceSpan(ordinal - 1, max(0, previous - radius), previous))
    spans.append(SourceSpan(ordinal, max(0, focus.start - radius), min(size, focus.end + radius)))
    if focus.end == size and ordinal + 1 < len(messages) and radius:
        following = len(messages[ordinal + 1].content)
        if following:
            spans.append(SourceSpan(ordinal + 1, 0, min(radius, following)))
    return tuple(spans)


def context_cost(context: tuple[SourceSpan, ...]) -> int:
    """Per-window delivered code points; overlap across windows is charged again."""
    return sum(span.end - span.start for span in context)
