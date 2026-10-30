"""Explicit local response replay. This adapter performs no inference or network I/O."""

from dataclasses import dataclass

from memory_optimizer.domain import digest
from memory_optimizer.extraction.json_codec import canonical, strict_json
from memory_optimizer.extraction.types import ExtractionInput, ExtractionReply
from memory_optimizer.extraction.validation import array, choice, fields, text


@dataclass(frozen=True)
class ReplayExtractor:
    source_payload_hash: str
    responses: tuple[tuple[int, tuple[ExtractionReply, ...]], ...]

    @classmethod
    def from_json(cls, content: str):
        manifest = fields(strict_json(content, 1024 * 1024), "source_payload_hash windows")
        source_hash = text(manifest["source_payload_hash"], 64)
        if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("invalid replay source fingerprint")
        windows = []
        seen = set()
        for raw in array(manifest["windows"], 256):
            item = fields(raw, "ordinal replies")
            ordinal = item["ordinal"]
            if type(ordinal) is not int or not 0 <= ordinal < 256 or ordinal in seen:
                raise ValueError("invalid replay window ordinal")
            seen.add(ordinal)
            replies = []
            for raw_reply in array(item["replies"], 2, 1):
                reply = fields(raw_reply, "status content")
                status = choice(reply["status"], "complete truncated refused error")
                if reply["content"] is not None and type(reply["content"]) is not str:
                    raise ValueError("invalid replay response")
                replies.append(ExtractionReply(status, reply["content"]))
            windows.append((ordinal, tuple(replies)))
        return cls(source_hash, tuple(sorted(windows)))

    @property
    def version(self) -> str:
        from dataclasses import asdict

        return "replay-v1:" + digest(canonical(asdict(self)))

    def extract(self, request: ExtractionInput, attempt: int) -> ExtractionReply:
        if request.payload_hash != self.source_payload_hash:
            return ExtractionReply("error")
        replies = dict(self.responses).get(request.window.ordinal, ())
        return replies[attempt - 1] if 1 <= attempt <= len(replies) else ExtractionReply("error")
