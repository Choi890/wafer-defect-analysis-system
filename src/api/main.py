from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.database.repository import bootstrap_database


app = FastAPI(
    title="Wafer Defect Analysis System API",
    description="FastAPI backend for wafer defect predictions, metrics, and quality statistics.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    bootstrap_database()


app.include_router(router)
