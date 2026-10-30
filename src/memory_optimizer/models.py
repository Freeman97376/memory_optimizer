"""Fixed competition boundary; no coercion or silent field loss."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Text = Annotated[str, StringConstraints(strict=True, min_length=1)]


def postgres_text(value: str) -> str:
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("text must be valid Unicode without NUL or surrogate code points")
    return value


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def check_text(cls, value):
        return postgres_text(value) if isinstance(value, str) else value


class Message(Contract):
    role: Literal["user", "assistant"]
    content: Text
    timestamp: Annotated[int, Field(ge=-(2**63), le=2**63 - 1)] | None = None


class AddRequest(Contract):
    request_id: Text
    user_id: Text
    session_id: Text
    messages: Annotated[list[Message], Field(min_length=1)]


class AddResponse(Contract):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(Contract):
    query: Text
    user_id: Text
    top_k: Annotated[int, Field(ge=1, le=100)] = 100
    options: list[str] | None = None


class SearchHit(Contract):
    id: Text
    content: Text
    score: float


class SearchResponse(Contract):
    data: list[SearchHit]
