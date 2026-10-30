"""Small immutable call contracts. Money uses integer nanodollars."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

MODEL = "gpt-4o-mini-2024-07-18"
CAMPAIGN = "amc-20261029-preflight"
INPUT_NUSD = 150
OUTPUT_NUSD = 600
CONTEXT_TOKENS = 128000


@dataclass(frozen=True)
class Limits:
    campaign: str = CAMPAIGN
    max_calls: int = 200
    max_nusd: int = 10_000_000_000

    def __post_init__(self):
        if not self.campaign or len(self.campaign) > 100:
            raise ValueError("invalid campaign")
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 200:
            raise ValueError("call limit exceeds approved maximum")
        if type(self.max_nusd) is not int or not 1 <= self.max_nusd <= 10_000_000_000:
            raise ValueError("cost limit exceeds approved maximum")


@dataclass(frozen=True)
class Result:
    status: str
    content: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    @property
    def known_cost(self):
        if type(self.input_tokens) is int and type(self.output_tokens) is int:
            if 0 <= self.input_tokens <= CONTEXT_TOKENS and 0 <= self.output_tokens <= 16384:
                return self.input_tokens * INPUT_NUSD + self.output_tokens * OUTPUT_NUSD
        return None


@dataclass(frozen=True)
class Reservation:
    id: str
    reserved_nusd: int
    expires_at: datetime


class CallBlocked(Exception):
    """A closed call gate. No network operation is permitted."""


class Ledger(Protocol):
    def reserve(
        self, limits: Limits, scope: str, key: str, deadline: datetime, max_output: int
    ) -> Reservation | Result: ...
    def finish(self, reservation: Reservation, result: Result) -> Result: ...
    def summary(self, campaign: str) -> dict: ...


class Transport(Protocol):
    async def complete(self, body: dict, timeout: float) -> Result: ...
