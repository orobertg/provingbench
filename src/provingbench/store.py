"""SQLite: every model call, what it was flagged with, which model each agent uses, and test results."""
import contextlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY, upstream TEXT NOT NULL, price_prompt REAL, price_completion REAL,
    context_length INTEGER, supports_tools INTEGER, seen_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, last_live_at TEXT,
    allowed_models TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS selections (
    id INTEGER PRIMARY KEY, agent_id TEXT NOT NULL, model TEXT, by_whom TEXT NOT NULL, at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS selections_agent ON selections (agent_id, id);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, agent_id TEXT NOT NULL, model TEXT NOT NULL, upstream TEXT NOT NULL,
    state TEXT NOT NULL, status INTEGER, request TEXT NOT NULL, response TEXT NOT NULL, tool_calls TEXT NOT NULL,
    finish TEXT, ttft_s REAL, total_s REAL, prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL,
    run_id INTEGER, task_id TEXT, attempt_id TEXT);
CREATE INDEX IF NOT EXISTS calls_attempt ON calls (attempt_id);
CREATE INDEX IF NOT EXISTS calls_run ON calls (run_id);
CREATE TABLE IF NOT EXISTS flags (call_id INTEGER NOT NULL, name TEXT NOT NULL, detail TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS flags_call ON flags (call_id);
CREATE INDEX IF NOT EXISTS flags_name ON flags (name);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, target TEXT NOT NULL, agent_id TEXT NOT NULL, suite TEXT NOT NULL,
    models TEXT NOT NULL, repeats INTEGER, max_cost_usd REAL NOT NULL, label TEXT NOT NULL, state TEXT NOT NULL,
    saved_selection TEXT, total INTEGER NOT NULL, error TEXT, ended_at TEXT);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, run_id INTEGER NOT NULL, model TEXT NOT NULL, task_id TEXT NOT NULL,
    task_hash TEXT NOT NULL, tags TEXT NOT NULL, repeat INTEGER NOT NULL, attempt_id TEXT NOT NULL,
    outcome TEXT NOT NULL, checks TEXT NOT NULL, reply TEXT, error TEXT, seconds REAL, cost_usd REAL);
CREATE INDEX IF NOT EXISTS results_run ON results (run_id);
CREATE TABLE IF NOT EXISTS overrides (
    id INTEGER PRIMARY KEY, result_id INTEGER NOT NULL, outcome TEXT NOT NULL, note TEXT NOT NULL, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (model_id TEXT PRIMARY KEY, text TEXT NOT NULL, at TEXT NOT NULL);
PRAGMA user_version = 1;
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Flag:
    name: str
    detail: str = ""


@dataclass(frozen=True)
class CallRecord:
    agent_id: str
    model: str
    upstream: str
    state: str              # ok | cancelled | error
    status: int | None
    request: dict
    response: dict
    tool_calls: list[dict]  # {id, name, arguments, answered}; answered is real | stubbed | blocked
    finish: str | None
    ttft_s: float | None
    total_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    flags: list[Flag]
    run_id: int | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    at: str | None = None


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=1)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.log_errors = 0

    def close(self) -> None:
        self.conn.close()

    @contextlib.contextmanager
    def write(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # Agents and what they use

    def _ensure_agent(self, conn, agent_id: str) -> None:
        at = now_iso()
        conn.execute("INSERT INTO agents (id, first_seen, last_seen) VALUES (?, ?, ?) "
                     "ON CONFLICT (id) DO UPDATE SET last_seen = excluded.last_seen", (agent_id, at, at))

    def touch_agent(self, agent_id: str, live: bool) -> None:
        with self.write() as conn:
            self._ensure_agent(conn, agent_id)
            if live:
                conn.execute("UPDATE agents SET last_live_at = ? WHERE id = ?", (now_iso(), agent_id))

    def agents(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM agents ORDER BY id").fetchall()
        return [{**dict(r), "allowed_models": json.loads(r["allowed_models"]), "model": self.selection(r["id"])}
                for r in rows]

    def last_live_at(self, agent_id: str) -> str | None:
        row = self.conn.execute("SELECT last_live_at FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return row["last_live_at"] if row else None

    def selection(self, agent_id: str) -> str | None:
        row = self.conn.execute("SELECT model FROM selections WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
                                (agent_id,)).fetchone()
        return row["model"] if row else None

    def select(self, agent_id: str, model: str | None, by: str) -> None:
        with self.write() as conn:
            self._ensure_agent(conn, agent_id)
            conn.execute("INSERT INTO selections (agent_id, model, by_whom, at) VALUES (?, ?, ?, ?)",
                         (agent_id, model, by, now_iso()))

    def selections(self, agent_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT model, by_whom, at FROM selections WHERE agent_id = ? ORDER BY id",
                                 (agent_id,)).fetchall()
        return [{"model": r["model"], "by": r["by_whom"], "at": r["at"]} for r in rows]

    def allowed_models(self, agent_id: str) -> list[str]:
        row = self.conn.execute("SELECT allowed_models FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return json.loads(row["allowed_models"]) if row else []

    def set_allowed(self, agent_id: str, models: list[str]) -> None:
        with self.write() as conn:
            self._ensure_agent(conn, agent_id)
            conn.execute("UPDATE agents SET allowed_models = ? WHERE id = ?", (json.dumps(models), agent_id))

    # Models

    def save_models(self, rows: list[dict]) -> None:
        at = now_iso()
        with self.write() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO models (id, upstream, price_prompt, price_completion, context_length, "
                "supports_tools, seen_at) VALUES (:id, :upstream, :price_prompt, :price_completion, :context_length, "
                f":supports_tools, '{at}')", rows)

    def models(self) -> list[dict]:
        return [_model(r) for r in self.conn.execute("SELECT * FROM models ORDER BY id").fetchall()]

    def model(self, model_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return _model(row) if row else None

    # Calls

    def add_call(self, rec: CallRecord) -> int:
        with self.write() as conn:
            cur = conn.execute(
                "INSERT INTO calls (at, agent_id, model, upstream, state, status, request, response, tool_calls, finish, "
                "ttft_s, total_s, prompt_tokens, completion_tokens, cost_usd, run_id, task_id, attempt_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rec.at or now_iso(), rec.agent_id, rec.model, rec.upstream, rec.state, rec.status,
                 json.dumps(rec.request), json.dumps(rec.response), json.dumps(rec.tool_calls), rec.finish, rec.ttft_s,
                 rec.total_s, rec.prompt_tokens, rec.completion_tokens, rec.cost_usd, rec.run_id, rec.task_id,
                 rec.attempt_id))
            conn.executemany("INSERT INTO flags (call_id, name, detail) VALUES (?, ?, ?)",
                             [(cur.lastrowid, f.name, f.detail) for f in rec.flags])
        return cur.lastrowid

    def _call(self, row) -> dict:
        flags = self.conn.execute("SELECT name, detail FROM flags WHERE call_id = ?", (row["id"],)).fetchall()
        return {**dict(row), "request": json.loads(row["request"]), "response": json.loads(row["response"]),
                "tool_calls": json.loads(row["tool_calls"]), "flags": [dict(f) for f in flags]}

    def call(self, call_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        return self._call(row) if row else None

    def calls(self, after: int = 0, agent: str | None = None, model: str | None = None, flag: str | None = None,
              limit: int = 100) -> list[dict]:
        """Calls in id order: the first `limit` after `after`, or with no `after` the latest `limit`."""
        where, args = ["id > ?"], [after]
        if agent:
            where.append("agent_id = ?")
            args.append(agent)
        if model:
            where.append("model = ?")
            args.append(model)
        if flag:
            where.append("id IN (SELECT call_id FROM flags WHERE name = ?)")
            args.append(flag)
        order = "ASC" if after else "DESC"
        rows = self.conn.execute(f"SELECT * FROM calls WHERE {' AND '.join(where)} ORDER BY id {order} LIMIT ?",
                                 (*args, limit)).fetchall()
        return [self._call(r) for r in sorted(rows, key=lambda r: r["id"])]

    def attempt_calls(self, attempt_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM calls WHERE attempt_id = ? ORDER BY id", (attempt_id,)).fetchall()
        return [self._call(r) for r in rows]

    def prune(self, before_iso: str) -> int:
        with self.write() as conn:
            conn.execute("DELETE FROM flags WHERE call_id IN (SELECT id FROM calls WHERE at < ?)", (before_iso,))
            return conn.execute("DELETE FROM calls WHERE at < ?", (before_iso,)).rowcount

    # Runs and results

    def create_run(self, target: str, agent_id: str, suite: str, models: list[str], repeats: int | None,
                   max_cost_usd: float, label: str, saved_selection: str | None, total: int) -> int:
        with self.write() as conn:
            return conn.execute(
                "INSERT INTO runs (at, target, agent_id, suite, models, repeats, max_cost_usd, label, state, "
                "saved_selection, total) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)",
                (now_iso(), target, agent_id, suite, json.dumps(models), repeats, max_cost_usd, label, saved_selection,
                 total)).lastrowid

    def update_run(self, run_id: int, **fields) -> None:
        names = ", ".join(f"{name} = ?" for name in fields)     # names come only from this package's own code
        with self.write() as conn:
            conn.execute(f"UPDATE runs SET {names} WHERE id = ?", (*fields.values(), run_id))

    def run(self, run_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return {**dict(row), "models": json.loads(row["models"])} if row else None

    def runs(self, state: str | None = None, limit: int = 100) -> list[dict]:
        where, args = ("WHERE state = ?", (state,)) if state else ("", ())
        rows = self.conn.execute(f"SELECT * FROM runs {where} ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [{**dict(r), "models": json.loads(r["models"])} for r in rows]

    def add_result(self, run_id: int, model: str, task_id: str, task_hash: str, tags: list[str], repeat: int,
                   attempt_id: str, outcome: str, checks: list[dict], reply: str | None, error: str | None,
                   seconds: float, cost_usd: float) -> int:
        with self.write() as conn:
            return conn.execute(
                "INSERT INTO results (at, run_id, model, task_id, task_hash, tags, repeat, attempt_id, outcome, checks, "
                "reply, error, seconds, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now_iso(), run_id, model, task_id, task_hash, json.dumps(tags), repeat, attempt_id, outcome,
                 json.dumps(checks), reply, error, seconds, cost_usd)).lastrowid

    def results(self, run_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        return [_result(r) for r in rows]

    def result(self, result_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
        return _result(row) if row else None

    def done_attempts(self, run_id: int) -> set[tuple]:
        rows = self.conn.execute("SELECT model, task_id, repeat FROM results WHERE run_id = ?", (run_id,)).fetchall()
        return {(r["model"], r["task_id"], r["repeat"]) for r in rows}

    def spend(self, run_id: int) -> float:
        row = self.conn.execute("SELECT SUM(cost_usd) AS total FROM calls WHERE run_id = ?", (run_id,)).fetchone()
        return row["total"] or 0.0

    def all_results(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM results ORDER BY id").fetchall()
        return [_result(r) for r in rows]

    def add_override(self, result_id: int, outcome: str, note: str) -> None:
        with self.write() as conn:
            conn.execute("INSERT INTO overrides (result_id, outcome, note, at) VALUES (?, ?, ?, ?)",
                         (result_id, outcome, note, now_iso()))

    def overrides(self) -> dict[int, dict]:
        """The latest override of each result that has one."""
        rows = self.conn.execute("SELECT result_id, outcome, note, at FROM overrides ORDER BY id").fetchall()
        return {r["result_id"]: {"outcome": r["outcome"], "note": r["note"], "at": r["at"]} for r in rows}

    def note(self, model_id: str, text: str) -> None:
        with self.write() as conn:
            conn.execute("INSERT OR REPLACE INTO notes (model_id, text, at) VALUES (?, ?, ?)", (model_id, text, now_iso()))

    def notes(self, model_id: str) -> str:
        row = self.conn.execute("SELECT text FROM notes WHERE model_id = ?", (model_id,)).fetchone()
        return row["text"] if row else ""

    def model_flags(self, model_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT f.name AS name, COUNT(*) AS count FROM flags f JOIN calls c ON c.id = f.call_id WHERE c.model = ? "
            "GROUP BY f.name ORDER BY count DESC, name", (model_id,)).fetchall()
        return [dict(r) for r in rows]

    def model_ttfts(self, model_id: str) -> list[float]:
        rows = self.conn.execute("SELECT ttft_s FROM calls WHERE model = ? AND ttft_s IS NOT NULL", (model_id,)).fetchall()
        return [r["ttft_s"] for r in rows]


def _result(row) -> dict:
    return {**dict(row), "tags": json.loads(row["tags"]), "checks": json.loads(row["checks"])}


def _model(row) -> dict:
    model = dict(row)
    if model["supports_tools"] is not None:
        model["supports_tools"] = bool(model["supports_tools"])
    return model
