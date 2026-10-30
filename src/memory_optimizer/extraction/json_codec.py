"""Strict bounded JSON boundary; errors contain codes, never source/output values."""

import json

from memory_optimizer.extraction.types import CandidateValidationError


def canonical(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def strict_json(text: str, limit: int):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CandidateValidationError("duplicate_json_key")
            result[key] = value
        return result

    def non_finite(value):
        raise CandidateValidationError("invalid_json")

    try:
        if type(text) is not str or len(text.encode("utf-8")) > limit:
            raise CandidateValidationError("response_size")
        # Bound nesting before entering the recursive JSON decoder. Quoted braces
        # and escaped quotes are data and must not contribute to structural depth.
        depth, quoted, escaped = 0, False, False
        for character in text:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "[{":
                depth += 1
                if depth > 64:
                    raise CandidateValidationError("invalid_json")
            elif character in "]}":
                depth -= 1
        return json.loads(text, object_pairs_hook=object_pairs, parse_constant=non_finite)
    except (UnicodeError, RecursionError, json.JSONDecodeError):
        raise CandidateValidationError("invalid_json") from None
