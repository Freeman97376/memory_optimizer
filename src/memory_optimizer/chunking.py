"""Lossless source slices; offsets refer to Unicode code points, not bytes."""

from collections.abc import Iterator

from memory_optimizer.domain import SourceSlice


def chunks(content: str, size: int, overlap: int) -> Iterator[SourceSlice]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("invalid chunk size or overlap")
    start = 0
    ordinal = 0
    while start < len(content):
        end = min(start + size, len(content))
        yield SourceSlice(ordinal, start, end, content[start:end])
        if end == len(content):
            break
        start = end - overlap
        ordinal += 1
