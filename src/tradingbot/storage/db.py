"""Engine, sessions and automatic schema creation for the live-state SQLite file."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from tradingbot.config.models import AppConfig
from tradingbot.core.exceptions import StorageError
from tradingbot.storage.models import Base

SCHEMA_VERSION = 1
MEMORY_URL = "sqlite:///:memory:"


def sqlite_url(path: Path | str) -> str:
    """Turn a filesystem path, ``:memory:``, or an existing URL into a SQLAlchemy URL."""
    raw = str(path)
    if raw in {":memory:", MEMORY_URL}:
        return MEMORY_URL
    if raw.startswith("sqlite:"):
        return raw
    resolved = Path(raw).expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(f"cannot create directory for state db {resolved}") from exc
    return f"sqlite:///{resolved}"


def _configure_sqlite(engine: Engine) -> None:
    """Foreign keys and WAL are cheap insurance on a single-writer bot process."""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


class Database:
    """SQLite handle. The schema is created on construction (DoD: automatic init)."""

    def __init__(self, url_or_path: Path | str = ":memory:", *, echo: bool = False) -> None:
        self.url = sqlite_url(url_or_path)
        kwargs: dict[str, Any] = {"echo": echo, "future": True}
        if self.url == MEMORY_URL:
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = StaticPool
        else:
            kwargs["connect_args"] = {"check_same_thread": False}
        try:
            self.engine = create_engine(self.url, **kwargs)
        except Exception as exc:
            raise StorageError(f"cannot open state db {self.url}") from exc
        if self.url != MEMORY_URL:
            _configure_sqlite(self.engine)
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        self.create_schema()

    @classmethod
    def from_config(cls, config: AppConfig, *, echo: bool = False) -> Database:
        """Open the database path from ``live.state_db``."""
        return cls(config.live.state_db, echo=echo)

    def create_schema(self) -> None:
        """Create missing tables and stamp the schema version."""
        Base.metadata.create_all(self.engine)
        from tradingbot.storage.repositories import StateRepo

        with self.session_scope() as session:
            StateRepo(session).ensure_schema_version(SCHEMA_VERSION)
        logger.debug(
            "state db ready at {url} schema v{version}",
            url=self.url,
            version=SCHEMA_VERSION,
        )

    def session(self) -> Session:
        """A new session. Prefer :meth:`session_scope` so commits and rollbacks stay paired."""
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Commit on success, roll back on error, always close."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Release pooled connections."""
        self.engine.dispose()


def open_database(path: Path | str = ":memory:", *, echo: bool = False) -> Database:
    """Public constructor used by the live runner and tests."""
    return Database(path, echo=echo)
