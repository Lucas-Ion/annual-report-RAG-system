"""Progress tracking for the four ingest stages."""

from __future__ import annotations

import sqlite3

from app.db.models import Stage, StageRun, StageStatus
from app.db.repositories.base import Repository

_COLUMNS = "document_id, stage, status, started_at, finished_at, error"


def to_stage_run(row: sqlite3.Row) -> StageRun:
    return StageRun(
        document_id=row["document_id"],
        stage=Stage(row["stage"]),
        status=StageStatus(row["status"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error=row["error"],
    )


class StageRunRepository(Repository[StageRun, tuple[int, Stage]]):
    def read(self) -> list[StageRun]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM stage_runs ORDER BY document_id, stage"
        ).fetchall()
        return [to_stage_run(row) for row in rows]

    def read_by_id(self, entity_id: tuple[int, Stage]) -> StageRun | None:
        document_id, stage = entity_id
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM stage_runs WHERE document_id = ? AND stage = ?",
            (document_id, str(stage)),
        ).fetchone()
        return to_stage_run(row) if row else None

    def read_for_document(self, document_id: int) -> list[StageRun]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM stage_runs WHERE document_id = ? ORDER BY stage",
            (document_id,),
        ).fetchall()
        return [to_stage_run(row) for row in rows]

    def is_done(self, document_id: int, stage: Stage) -> bool:
        run = self.read_by_id((document_id, stage))
        return run is not None and run.status is StageStatus.DONE

    def start(self, document_id: int, stage: Stage) -> StageRun:
        return self._write(
            StageRun(
                document_id=document_id,
                stage=stage,
                status=StageStatus.RUNNING,
            ),
            touch_started=True,
        )

    def finish(self, document_id: int, stage: Stage) -> StageRun:
        existing = self.read_by_id((document_id, stage))
        return self._write(
            StageRun(
                document_id=document_id,
                stage=stage,
                status=StageStatus.DONE,
                started_at=existing.started_at if existing else None,
            ),
            touch_finished=True,
        )

    def fail(self, document_id: int, stage: Stage, error: str) -> StageRun:
        existing = self.read_by_id((document_id, stage))
        return self._write(
            StageRun(
                document_id=document_id,
                stage=stage,
                status=StageStatus.FAILED,
                started_at=existing.started_at if existing else None,
                error=error,
            ),
            touch_finished=True,
        )

    def create(self, entity: StageRun) -> StageRun:
        return self._write(entity)

    def update(self, entity: StageRun) -> StageRun:
        return self._write(entity)

    def delete(self, entity: StageRun) -> StageRun:
        self._conn.execute(
            "DELETE FROM stage_runs WHERE document_id = ? AND stage = ?",
            (entity.document_id, str(entity.stage)),
        )
        return entity

    def _write(
        self,
        entity: StageRun,
        *,
        touch_started: bool = False,
        touch_finished: bool = False,
    ) -> StageRun:
        """Insert or replace a stage record.

        SQLite's ON CONFLICT DO UPDATE is used rather than INSERT OR REPLACE.
        The latter deletes the old row before inserting the new one, which
        would fire delete triggers and cascades. Nothing depends on stage_runs
        today, so it would not currently break anything, but relying on that
        staying true is not worth the two extra lines.

        Args:
            entity: The record to store.
            touch_started: Set started_at to now and clear finished_at and
                error. Used when a stage begins.
            touch_finished: Set finished_at to now. Used when a stage ends,
                whether it succeeded or not.

        Returns:
            The record as it now stands in the database.
        """
        started = "datetime('now')" if touch_started else "?"
        finished = "datetime('now')" if touch_finished else "?"
        values: list[object] = [
            entity.document_id,
            str(entity.stage),
            str(entity.status),
        ]
        if not touch_started:
            values.append(entity.started_at)
        if not touch_finished:
            values.append(entity.finished_at)
        values.append(None if touch_started else entity.error)

        self._conn.execute(
            f"""
            INSERT INTO stage_runs (
                document_id, stage, status, started_at, finished_at, error
            )
            VALUES (?, ?, ?, {started}, {finished}, ?)
            ON CONFLICT (document_id, stage) DO UPDATE SET
                status      = excluded.status,
                started_at  = excluded.started_at,
                finished_at = excluded.finished_at,
                error       = excluded.error
            """,
            values,
        )
        stored = self.read_by_id((entity.document_id, entity.stage))
        assert stored is not None  # just written, inside the same transaction
        return stored
