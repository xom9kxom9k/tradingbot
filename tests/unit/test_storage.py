"""SQLite live-state store: schema, repositories and signal idempotency."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from tradingbot.config.models import AppConfig, ExchangeConfig, LiveConfig
from tradingbot.core.enums import ExitReason, PositionStatus, Side, SignalStatus, SignalType
from tradingbot.core.models import Position, Signal, TakeProfit, Trade
from tradingbot.storage import SCHEMA_VERSION, Database, bind_repos, open_database
from tradingbot.storage.models import PositionRow, SignalRow
from tradingbot.storage.repositories import SignalRepo, StateRepo

TS = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def make_signal(**overrides: object) -> Signal:
    values: dict[str, object] = {
        "signal_type": SignalType.ENTRY,
        "side": Side.LONG,
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "bar_ts": TS,
        "price": 63_250.0,
        "stop_loss": 61_100.0,
        "size": 0.145,
        "risk_pct": 0.01,
        "reason": "donchian breakout",
        "take_profits": [TakeProfit(price=67_550.0, r_multiple=2.0, fraction=0.5)],
        "indicators": {"adx": 28.4, "atr": 860.0},
    }
    values.update(overrides)
    return Signal(**values)  # type: ignore[arg-type]


def make_position(**overrides: object) -> Position:
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "side": Side.LONG,
        "entry_ts": TS,
        "entry_price": 63_250.0,
        "size": 0.145,
        "initial_size": 0.145,
        "initial_stop": 61_100.0,
        "current_stop": 61_100.0,
        "highest_price": 63_250.0,
        "lowest_price": 63_200.0,
        "signal_uid": make_signal().uid,
        "equity_at_entry": 10_000.0,
    }
    values.update(overrides)
    return Position(**values)  # type: ignore[arg-type]


def make_trade(**overrides: object) -> Trade:
    values: dict[str, object] = {
        "trade_id": "11111111-1111-4111-8111-111111111111",
        "symbol": "BTC/USDT",
        "side": Side.LONG,
        "entry_ts": TS,
        "entry_price": 63_250.0,
        "exit_ts": TS + timedelta(hours=8),
        "exit_price": 64_000.0,
        "size": 0.145,
        "initial_stop": 61_100.0,
        "r_value": 2_150.0,
        "pnl_gross": 108.75,
        "fees": 1.2,
        "funding": 0.3,
        "pnl_net": 107.25,
        "pnl_r": 0.35,
        "pnl_pct": 0.0107,
        "bars_held": 2,
        "exit_reason": ExitReason.TP1,
        "mae": -0.2,
        "mfe": 0.8,
        "entry_reason": "donchian breakout",
    }
    values.update(overrides)
    return Trade(**values)  # type: ignore[arg-type]


@pytest.fixture
def db() -> Database:
    return open_database(":memory:")


class TestSchema:
    def test_all_tables_are_created(self, db: Database) -> None:
        names = set(inspect(db.engine).get_table_names())
        assert names >= {
            "signals",
            "positions",
            "trades",
            "equity_snapshots",
            "bot_state",
            "notification_queue",
        }

    def test_signal_uid_has_a_unique_index(self, db: Database) -> None:
        inspector = inspect(db.engine)
        unique_indexes = [item for item in inspector.get_indexes("signals") if item.get("unique")]
        unique_constraints = inspector.get_unique_constraints("signals")
        covered = any("signal_uid" in (item.get("column_names") or []) for item in unique_indexes)
        covered = covered or any(
            "signal_uid" in item["column_names"] for item in unique_constraints
        )
        assert covered

    def test_schema_version_is_stamped(self, db: Database) -> None:
        with db.session_scope() as session:
            assert StateRepo(session).schema_version() == SCHEMA_VERSION

    def test_file_database_is_created_automatically(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "state.db"
        store = Database(path)
        try:
            assert path.is_file()
            with store.session_scope() as session:
                assert StateRepo(session).schema_version() == SCHEMA_VERSION
        finally:
            store.dispose()

    def test_from_config_uses_live_state_db(self, tmp_path: Path) -> None:
        config = AppConfig(
            exchange=ExchangeConfig(symbols=["BTC/USDT"]),
            live=LiveConfig(state_db=tmp_path / "bot.db"),
        )
        store = Database.from_config(config)
        try:
            assert (tmp_path / "bot.db").is_file()
        finally:
            store.dispose()


class TestSignalIdempotency:
    def test_second_insert_of_the_same_uid_does_not_create_a_duplicate(self, db: Database) -> None:
        signal = make_signal()
        with db.session_scope() as session:
            repo = SignalRepo(session)
            first, created_first = repo.save(signal)
            second, created_second = repo.save(signal)
            assert created_first is True
            assert created_second is False
            assert first.id == second.id
            assert repo.count() == 1
            assert second.signal_uid == signal.uid

    def test_unique_index_rejects_a_raw_duplicate_row(self, db: Database) -> None:
        signal = make_signal()
        with db.session_scope() as session:
            session.add(SignalRow.from_signal(signal))
        with pytest.raises(IntegrityError), db.session_scope() as session:
            session.add(SignalRow.from_signal(signal))
            session.flush()

    def test_round_trip_preserves_take_profits_and_indicators(self, db: Database) -> None:
        signal = make_signal()
        with db.session_scope() as session:
            row, _ = SignalRepo(session).save(signal)
            restored = row.to_signal()
        assert restored.uid == signal.uid
        assert restored.take_profits[0].r_multiple == pytest.approx(2.0)
        assert restored.indicators["adx"] == pytest.approx(28.4)
        assert restored.bar_ts.tzinfo is not None

    def test_mark_notified(self, db: Database) -> None:
        signal = make_signal()
        with db.session_scope() as session:
            repo = SignalRepo(session)
            repo.save(signal)
            row = repo.mark_notified(signal.uid)
            assert row is not None
            assert row.status == SignalStatus.NOTIFIED.value
            assert row.notified_at is not None


class TestPositions:
    def test_upsert_then_close(self, db: Database) -> None:
        position = make_position()
        with db.session_scope() as session:
            repos = bind_repos(session)
            repos.positions.upsert_open(position)
            opened = repos.positions.get_open("BTC/USDT")
            assert opened is not None
            domain = opened.to_position()
            domain.current_stop = 62_000.0
            domain.size = 0.07
            repos.positions.close("BTC/USDT", domain)
            assert repos.positions.get_open("BTC/USDT") is None
            closed = session.scalar(
                select(PositionRow).where(PositionRow.status == PositionStatus.CLOSED.value)
            )
            assert closed is not None
            assert closed.current_stop == pytest.approx(62_000.0)
            assert closed.size == pytest.approx(0.07)

    def test_two_open_positions_on_the_same_symbol_are_rejected(self, db: Database) -> None:
        session = db.session()
        try:
            session.add(PositionRow.from_position(make_position()))
            session.flush()
            session.add(PositionRow.from_position(make_position(entry_price=64_000.0)))
            with pytest.raises(IntegrityError):
                session.flush()
        finally:
            session.rollback()
            session.close()


class TestTrades:
    def test_add_is_idempotent_on_trade_id(self, db: Database) -> None:
        trade = make_trade()
        with db.session_scope() as session:
            repos = bind_repos(session)
            _, created = repos.trades.add(trade)
            _, created_again = repos.trades.add(trade)
            assert created is True
            assert created_again is False
            assert repos.trades.count() == 1
            restored = repos.trades.get(trade.trade_id)
            assert restored is not None
            assert restored.to_trade().pnl_net == pytest.approx(trade.pnl_net)


class TestStateAndEquity:
    def test_pause_and_last_bar(self, db: Database) -> None:
        with db.session_scope() as session:
            state = StateRepo(session)
            assert state.is_paused() is False
            state.set_paused(True)
            state.set_last_processed_bar("BTC/USDT", TS)
            state.set_started_at(TS)
        with db.session_scope() as session:
            state = StateRepo(session)
            assert state.is_paused() is True
            assert state.last_processed_bar("BTC/USDT") == TS
            assert state.started_at() == TS

    def test_equity_snapshot(self, db: Database) -> None:
        with db.session_scope() as session:
            state = StateRepo(session)
            state.record_equity(
                TS, balance=10_000.0, equity=9_800.0, open_positions=1, drawdown_pct=0.02
            )
            latest = state.latest_equity()
            assert latest is not None
            assert latest.equity == pytest.approx(9_800.0)
            assert latest.open_positions_count == 1


class TestNotificationQueue:
    def test_pending_then_sent(self, db: Database) -> None:
        with db.session_scope() as session:
            repos = bind_repos(session)
            first = repos.notifications.enqueue({"text": "hello", "chat_id": 1})
            repos.notifications.enqueue({"text": "later", "chat_id": 1})
            pending = repos.notifications.pending()
            assert len(pending) == 2
            repos.notifications.record_attempt(first.id, "timeout")
            repos.notifications.mark_sent(first.id)
            leftover = repos.notifications.pending()
            assert len(leftover) == 1
            assert leftover[0].payload["text"] == "later"
            sent = session.get(type(first), first.id)
            assert sent is not None
            assert sent.attempts == 1
            assert sent.last_error == "timeout"
            assert sent.sent_at is not None

    def test_survives_reopen_on_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "state.db"
        first = open_database(path)
        try:
            with first.session_scope() as session:
                bind_repos(session).notifications.enqueue({"text": "queued"})
        finally:
            first.dispose()
        second = open_database(path)
        try:
            with second.session_scope() as session:
                pending = bind_repos(session).notifications.pending()
                assert len(pending) == 1
                assert pending[0].payload["text"] == "queued"
        finally:
            second.dispose()


def test_pragma_foreign_keys_on_file_db(tmp_path: Path) -> None:
    store = Database(tmp_path / "state.db")
    try:
        with store.engine.connect() as connection:
            enabled = connection.execute(text("PRAGMA foreign_keys")).scalar()
        assert int(enabled or 0) == 1
    finally:
        store.dispose()
