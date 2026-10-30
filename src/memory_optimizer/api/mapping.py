"""Translate the competition boundary without normalizing source data."""

from memory_optimizer.domain import IngestCommand, SearchQuery, SourceMessage
from memory_optimizer.models import AddRequest, SearchRequest


def ingest_command(request: AddRequest) -> IngestCommand:
    return IngestCommand(
        request_id=request.request_id,
        user_id=request.user_id,
        session_id=request.session_id,
        messages=tuple(
            SourceMessage(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in request.messages
        ),
    )


def search_query(request: SearchRequest) -> SearchQuery:
    return SearchQuery(
        query=request.query,
        user_id=request.user_id,
        top_k=request.top_k,
        options=None if request.options is None else tuple(request.options),
    )
