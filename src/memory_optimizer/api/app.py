from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from memory_optimizer.api.guard import RequestGuard
from memory_optimizer.api.mapping import ingest_command, search_query
from memory_optimizer.application import Services
from memory_optimizer.config import Settings
from memory_optimizer.domain import ProcessingUnavailable, RequestConflict, StorageUnavailable
from memory_optimizer.models import (
    AddRequest,
    AddResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate_service()
    if services is None:
        from memory_optimizer.bootstrap import build_services

        services = build_services(settings)

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(services.lifecycle.open)
        try:
            yield
        finally:
            await run_in_threadpool(services.lifecycle.close)

    app = FastAPI(
        title="Memory Optimizer",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )
    app.state.services = services
    app.add_middleware(RequestGuard, settings=settings)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        known_fields = {
            "body",
            "request_id",
            "user_id",
            "session_id",
            "messages",
            "role",
            "content",
            "timestamp",
            "query",
            "top_k",
            "options",
        }
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "details": [
                    {
                        "location": [
                            part if isinstance(part, int) or part in known_fields else "<unknown>"
                            for part in error["loc"]
                        ],
                        "type": error["type"],
                    }
                    for error in exc.errors()
                ],
            },
        )

    @app.exception_handler(RequestConflict)
    async def conflict(request: Request, exc: RequestConflict):
        return JSONResponse(status_code=409, content={"error": "request_id_conflict"})

    @app.exception_handler(StorageUnavailable)
    async def storage_unavailable(request: Request, exc: StorageUnavailable):
        return JSONResponse(
            status_code=503, content={"error": "storage_unavailable"}, headers={"Retry-After": "1"}
        )

    @app.exception_handler(ProcessingUnavailable)
    async def processing_unavailable(request: Request, exc: ProcessingUnavailable):
        return JSONResponse(
            status_code=503,
            content={"error": "processing_unavailable"},
            headers={"Retry-After": "1"},
        )

    @app.get("/health")
    def health():
        try:
            services.lifecycle.check_ready()
        except StorageUnavailable:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return {"status": "ok"}

    @app.post("/add", response_model=AddResponse)
    def add(request: AddRequest):
        receipt = services.add.add(ingest_command(request))
        return AddResponse(
            request_id=receipt.request_id, user_id=receipt.user_id, session_id=receipt.session_id
        )

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        return SearchResponse(
            data=[
                SearchHit(id=hit.id, content=hit.content, score=hit.score)
                for hit in services.search.search(search_query(request))
            ]
        )

    return app
