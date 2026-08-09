"""Progress tracking for the four ingest stages.

This is the smallest repository in the package and the one that earns its keep
fastest. Parsing a 434 page report takes around 45 minutes, and without a
record of what has already succeeded, any interruption means starting over.
"""

from __future__ import annotations

import sqlite3

from app.db.models import Stage, StageRun, StageStatus
from app.db.repositories.base import Repository

_COLUMNS = "document_id, stage, status, started_at, finished_at, error"


def to_stage_run(row: sqlite3.Row) -> StageRun:
    """Build a StageRun from a database row.

    Args:
        row: A row selected with _COLUMNS.

    Returns:
        The equivalent domain object.
    """
    return StageRun(
        document_id=row["document_id"],
        stage=Stage(row["stage"]),
        status=StageStatus(row["status"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error=row["error"],
    )


class StageRunRepository(Repository[StageRun, tuple[int, Stage]]):
    """Which ingest stages have run, for which documents, and how they went.

    The identity here is a pair rather than a row id, because a document has
    exactly one run of each stage. Starting a stage again overwrites the
    previous record instead of adding a second one, which keeps the question
    "can I skip this stage?" a single lookup with no ordering to reason about.

    The three lifecycle methods (start, finish, fail) are the interesting part
    of this class. create() and update() exist to satisfy the base contract
    and both funnel into the same upsert, because for a row keyed on a pair
    there is no meaningful difference between inserting and replacing.
    """

    def read(self) -> list[StageRun]:
        """Return every stage record for every document.

        Returns:
            All stage runs, grouped by document.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM stage_runs ORDER BY document_id, stage"
        ).fetchall()
        return [to_stage_run(row) for row in rows]

    def read_by_id(self, entity_id: tuple[int, Stage]) -> StageRun | None:
        """Look up one stage record.

        Args:
            entity_id: The (document_id, stage) pair to look for.

        Returns:
            The stage run, or None if that stage has never been started.
        """
        document_id, stage = entity_id
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM stage_runs WHERE document_id = ? AND stage = ?",
            (document_id, str(stage)),
        ).fetchone()
        return to_stage_run(row) if row else None

    def read_for_document(self, document_id: int) -> list[StageRun]:
        """Return every stage record for one document.

        Args:
            document_id: The document to report on.

        Returns:
            That document's stage runs. Usually between zero and four rows.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM stage_runs WHERE document_id = ? ORDER BY stage",
            (document_id,),
        ).fetchall()
        return [to_stage_run(row) for row in rows]

    def is_done(self, document_id: int, stage: Stage) -> bool:
        """Report whether a stage has already completed successfully.

        This is the question the pipeline asks before every stage, and it is
        what turns a resumed ingest into a fast no-op for the work that was
        already finished.

        Args:
            document_id: The document being processed.
            stage: The stage about to be attempted.

        Returns:
            True only if the stage has run and succeeded. A stage that failed,
            or one that was interrupted while still marked running, returns
            False so the pipeline attempts it again.
        """
        run = self.read_by_id((document_id, stage))
        return run is not None and run.status is StageStatus.DONE

    def start(self, document_id: int, stage: Stage) -> StageRun:
        """Mark a stage as running now.

        Clears any previous finish time and error, so a retry after a failure
        does not leave misleading remnants of the earlier attempt sitting in
        the row.

        Args:
            document_id: The document being processed.
            stage: The stage being attempted.

        Returns:
            The stored record.
        """
        return self._write(
            StageRun(
                document_id=document_id,
                stage=stage,
                status=StageStatus.RUNNING,
            ),
            touch_started=True,
        )

    def finish(self, document_id: int, stage: Stage) -> StageRun:
        """Mark a running stage as successfully completed.

        Args:
            document_id: The document being processed.
            stage: The stage that just finished.

        Returns:
            The stored record.
        """
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
        """Record that a stage failed, and why.

        The message is stored rather than only logged, because the interesting
        failures here happen 40 minutes into an unattended run. Having the
        reason sitting in the database next to the document is the difference
        between diagnosing it and reproducing it.

        Args:
            document_id: The document being processed.
            stage: The stage that failed.
            error: What went wrong. Usually repr() of the exception.

        Returns:
            The stored record.
        """
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
        """Store a stage record, replacing any existing one for the same pair.

        Args:
            entity: The record to store.

        Returns:
            The stored record.
        """
        return self._write(entity)

    def update(self, entity: StageRun) -> StageRun:
        """Store a stage record. Identical to create() for this repository.

        Args:
            entity: The record to store.

        Returns:
            The stored record.
        """
        return self._write(entity)

    def delete(self, entity: StageRun) -> StageRun:
        """Forget a stage record, so the pipeline will run that stage again.

        Useful when a stage's code has changed and its old result should no
        longer be trusted.

        Args:
            entity: The record to remove.

        Returns:
            The record that was removed.
        """
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
