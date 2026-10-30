"""Explicit v3 additions to frozen v2 selection; signals are heuristics, not facts."""

import heapq
import re
from itertools import groupby

from memory_optimizer.domain import ProcessingUnavailable
from memory_optimizer.processing.plan_types import CAPABILITY_SELECTOR, PLANNING_SELECTOR
from memory_optimizer.processing.planning_rules import signal_blocks as legacy_blocks

_FEATURES = re.compile(
    r"\b(?:add(?:ed|ing)?|introduc(?:e|ed|es|ing)|implement(?:ed|s|ing)?|"
    r"enhanc(?:e|ed|es|ement)|remov(?:e|ed|es|ing)|deprecat(?:e|ed|es|ion)|"
    r"support(?:s|ed|ing)?|enable(?:d|s)?)\b|新增|增加|引入|实现|支持|启用|移除|弃用",
    re.IGNORECASE,
)


def signal_blocks(content, end, focus_chars, selector_version):
    if selector_version == PLANNING_SELECTOR:
        yield from legacy_blocks(content, end, focus_chars)
        return
    if selector_version != CAPABILITY_SELECTOR:
        raise ProcessingUnavailable("unsupported signal rules")
    features = (
        (match.start() // focus_chars, ("change",))
        for match in _FEATURES.finditer(content[:end])
        if end == len(content) or match.end() < end
    )
    merged = heapq.merge(legacy_blocks(content, end, focus_chars), features, key=lambda x: x[0])
    for block, group in groupby(merged, key=lambda x: x[0]):
        yield block, tuple(sorted({signal for _, signals in group for signal in signals}))
