"""Environment loading for the application's entry points."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_environment(path: Path | None = None) -> bool:
    target = path or ENV_PATH
    if not target.is_file():
        return False
    load_dotenv(target, override=False)
    return True


PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"


def resolve_pdf(filename: str, directory: Path = PDF_DIR) -> Path | None:
    root = directory.resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate if candidate.is_file() else None
