"""Explicit local responses; this adapter never makes a provider call."""

from dataclasses import dataclass

from memory_optimizer.domain import digest
from memory_optimizer.extraction.json_codec import canonical, strict_json
from memory_optimizer.extraction.validation import array, fields, text
from memory_optimizer.relations.contracts import REPLY_VERSION


@dataclass(frozen=True)
class ReplayRelations:
    source_payload_hash: str
    replies: tuple[str, ...]
    version: str

    @classmethod
    def from_json(cls, content):
        value = fields(strict_json(content, 1024 * 1024), "source_payload_hash replies")
        payload = text(value["source_payload_hash"], 64)
        if len(payload) != 64 or any(c not in "0123456789abcdef" for c in payload):
            raise ValueError("invalid relation replay payload hash")
        for r in array(value["replies"], 2, 1):
            if type(r) is not dict or {"schema_version", "input_hash"} & set(r):
                raise ValueError("relation transport fields are adapter-owned")
        replies = tuple(canonical(r) for r in array(value["replies"], 2, 1))
        return cls(payload, replies, "relations-replay:" + digest(canonical(value)))

    def propose(self, request, attempt):
        value = strict_json(self.replies[min(attempt - 1, len(self.replies) - 1)], 65536)
        # Bind only the transport identity; never invent or repair any proposal.
        return canonical(dict(value, schema_version=REPLY_VERSION, input_hash=request.fingerprint))


@dataclass(frozen=True)
class JointExtractor:
    extractor: object
    version: str

    def extract(self, request, attempt):
        return self.extractor.extract(request, attempt)


def processor_version(extractor, proposer):
    return "s3-replay:" + digest(
        canonical([extractor.version, proposer.version, "relation-input-v1"])
    )
