"""Exercise the actual CLI, TCP transport and abrupt service-process restart."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pytest

from .conftest import counts, request_data

pytestmark = pytest.mark.integration


@pytest.fixture
def process_logs():
    # Keep temporary logs inside this project, independent of shared Windows temp ACLs.
    cache = Path(__file__).resolve().parents[1] / ".cache"
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="process-smoke-", dir=cache) as folder:
        yield Path(folder)


@contextmanager
def running_service(settings, log_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = os.environ.copy()
    environment.update(
        {
            "MEMORY_DATABASE_URL": settings.database_url.get_secret_value(),
            "MEMORY_DATABASE_SCHEMA": settings.schema,
            "MEMORY_API_KEY": settings.api_key.get_secret_value(),
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "memory_optimizer", "serve", "--port", str(port)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False
            ) as client:
                deadline = time.monotonic() + 15
                while True:
                    if process.poll() is not None:
                        pytest.fail("service exited before readiness; inspect private test log")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        pytest.fail("service did not become ready within 15 seconds")
                    time.sleep(0.05)
                yield client
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_http_add_survives_actual_process_restart(db_settings, db, process_logs):
    auth = {"Authorization": "Bearer local-test-key"}
    query = {"user_id": "project-a", "query": "cache", "top_k": 100}
    with running_service(db_settings, process_logs / "first.log") as first:
        assert first.post("/add", json=request_data()).status_code == 401
        assert first.post("/add", json=request_data(), headers=auth).status_code == 200
        before = first.post("/search", json=query, headers=auth).json()
        assert len(before["data"]) == 2
    with running_service(db_settings, process_logs / "second.log") as second:
        assert second.post("/search", json=query, headers=auth).json() == before
        assert second.post("/add", json=request_data(), headers=auth).status_code == 200
        assert counts(db) == (1, 2, 2)
