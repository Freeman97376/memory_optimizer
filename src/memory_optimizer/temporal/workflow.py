"""Version dispatch only; the existing workflow owns deadlines and finite attempts."""

from memory_optimizer.processing.coverage_workflow import CoverageFlow
from memory_optimizer.temporal.proposals import FLOW_VERSION, validate_temporal_reply


class TemporalCoverageFlow(CoverageFlow):
    FLOW_VERSION = FLOW_VERSION
    validate_reply = staticmethod(validate_temporal_reply)
