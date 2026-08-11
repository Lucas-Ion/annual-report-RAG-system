"""The application factory and its entry point.

    uv run uvicorn app.main:app --reload
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


def asset(path: str) -> str:
    target = STATIC / path
    if not target.is_file():
        return f"/static/{path}"
    return f"/static/{path}?v={int(target.stat().st_mtime)}"


def create_app() -> FastAPI:
    load_environment()
    init_db().close()

    application = FastAPI(
        title="Annual report RAG",
        description=(
            "Ask questions about annual reports and get answers with checkable sources."
        ),
        version="0.1.0",
    )
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.globals["asset"] = asset
    application.state.templates = templates
    application.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    application.include_router(pages.router)
    application.include_router(api.router)
    application.include_router(uploads.router)

    @application.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, object]:
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
