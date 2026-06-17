from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from src.api.routes import router
from src.config import APP_NAME, APP_VERSION
from src.database.repository import bootstrap_database


app = FastAPI(
    title=f"{APP_NAME} API",
    description="FastAPI backend for wafer defect predictions, metrics, and quality statistics.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.on_event("startup")
def startup() -> None:
    bootstrap_database()


app.include_router(router)
