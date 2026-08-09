"""The application factory and its entry point.

    uv run uvicorn app.main:app --reload

Deliberately small. Its whole job is to load the environment, make sure the
database exists, wire up templates and static files, and attach the routers.
Anything with a decision in it lives somewhere that can be tested without an
HTTP client.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import load_environment
from app.db.connection import database_path, init_db
from app.routes import api, pages, uploads

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
TEMPLATES = HERE / "templates"


def create_app() -> FastAPI:
    """Build the application.

    Returns:
        A configured FastAPI instance.
    """
    load_environment()

    # Applying the schema on startup is what lets a fresh checkout run without
    # a setup step. Every statement is guarded by IF NOT EXISTS, so against the
    # seeded database that ships with the repository this does nothing at all.
    init_db().close()

    application = FastAPI(
        title="Annual report RAG",
        description=(
            "Ask questions about annual reports and get answers with checkable sources."
        ),
        version="0.1.0",
    )
    application.state.templates = Jinja2Templates(directory=str(TEMPLATES))
    application.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    application.include_router(pages.router)
    application.include_router(api.router)
    application.include_router(uploads.router)

    @application.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, object]:
        """Report that the process is up and where its database is.

        Returns:
            A small status object.
        """
        path = database_path()
        return {
            "ok": True,
            "database": str(path),
            "size_mb": round(path.stat().st_size / 1_000_000, 1)
            if path.exists()
            else 0,
        }

    return application


app = create_app()
