# ProvingBench

A stand-alone local evaluator for AI agents: model proxy, per-model log, test runner, dashboard.

## Docs
- Connecting an agent: `docs/connecting-agents.md`

## Setup
`python -m venv .venv`, then `.venv/Scripts/pip install -e .[dev]` (POSIX: `.venv/bin/pip`).

## Tests
`.venv/Scripts/python -m pytest` (live-model tests are marked `live`, skipped by default)

## Global Constraints
- No code here mentions or imports any particular agent. Tests use a fake agent and a fake upstream.
- Runtime dependencies: `httpx`, `starlette`, `uvicorn` only.
- Every SQLite write is `BEGIN IMMEDIATE`; WAL mode; dates ISO-8601 UTC.
- Files of 150–300 lines, 500 as a soft ceiling.
- Binds to `127.0.0.1:7300` by default. API keys come from environment variables and are never logged.
- A logging failure never fails a call.
- Tools are denied by default in test runs.
