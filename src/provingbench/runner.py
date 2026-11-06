"""Test runs: every model, task and repeat sent to a test instance of an agent, one at a time, graded and saved."""
import asyncio
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import anyio
import httpx

from provingbench.checks import Attempt, grade
from provingbench.intercept import ActiveRun
from provingbench.suite import SuiteError, Task, load_suite
from provingbench.target import DEFAULT_TIMEOUT_S, Target, TargetError, load_targets, send
from provingbench.upstreams import is_paid

log = logging.getLogger("provingbench")
LIVE_GUARD = timedelta(minutes=1)
# Rough sizes for the estimate shown before a run starts.
EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS, EST_CALLS = 2000, 300, 2
NO_CALLS = ("the agent answered, but no model call reached ProvingBench for {agent_id}. Check that this target points at "
            "a test copy of the agent whose base URL ends in /a/{agent_id}/v1. If it's pointed at the live agent, this "
            "turn really ran there.")
LOG_FAILED = "couldn't log this attempt's model calls, so it can't be graded"


class RunRefused(Exception):
    def __init__(self, message: str, conflict: bool = False):
        super().__init__(message)
        self.conflict = conflict


@dataclass(frozen=True)
class RunSpec:
    target: str
    suite: str
    models: list[str]
    repeats: int | None
    max_cost_usd: float
    label: str = ""


class RunManager:
    def __init__(self, store, cfg, active: dict, *, trigger_transport=None, judge=None, now=lambda: datetime.now(UTC)):
        self.store, self.cfg, self.active, self.judge, self.now = store, cfg, active, judge, now
        self._transport = trigger_transport
        self._tasks: dict[int, asyncio.Task] = {}
        self._stopping: set[int] = set()
        self._ended: dict[str, float] = {}      # agent -> until when its late calls are refused

    def _load(self, spec: RunSpec) -> tuple[Target, list[Task]]:
        try:
            targets = load_targets(self.cfg.data_dir / "targets")
        except TargetError as e:
            raise RunRefused(str(e)) from e
        if spec.target not in targets:
            raise RunRefused(f"no target named {spec.target!r}")
        if spec.suite not in self.cfg.suites:
            raise RunRefused(f"no suite named {spec.suite!r}")
        if not spec.models:
            raise RunRefused("pick at least one model")
        try:
            return targets[spec.target], load_suite(self.cfg.suites[spec.suite])
        except SuiteError as e:
            raise RunRefused(str(e)) from e

    def _repeats(self, task: Task, spec: RunSpec) -> int:
        return task.repeats or spec.repeats or self.cfg.repeats

    def _guard(self, agent_id: str) -> None:
        if agent_id in self.active:
            raise RunRefused(f"a run for {agent_id} is already going", conflict=True)
        seen = self.store.last_live_at(agent_id)
        if seen and self.now() - datetime.fromisoformat(seen) < LIVE_GUARD:
            raise RunRefused(f"{agent_id} had live traffic in the last minute. Runs need a separate test instance of "
                             "the agent, with its own agent ID.", conflict=True)

    def _launch(self, run_id: int, target: Target, tasks: list[Task], spec: RunSpec, saved: str | None) -> None:
        self._ended.pop(target.agent_id, None)
        self.active[target.agent_id] = ActiveRun(run_id, limit_usd=spec.max_cost_usd, grace_s=target.timeout_s)
        task = asyncio.create_task(self._run(run_id, target, tasks, spec, saved))
        self._tasks[run_id] = task
        # A task cancelled before its first step never runs its own cleanup.
        task.add_done_callback(lambda t: self._finish(run_id, target.agent_id, saved, self._ended_state(run_id), None))

    def start(self, spec: RunSpec) -> int:
        target, tasks = self._load(spec)
        self._guard(target.agent_id)
        saved = self.store.selection(target.agent_id)
        total = len(spec.models) * sum(self._repeats(task, spec) for task in tasks)
        run_id = self.store.create_run(spec.target, target.agent_id, spec.suite, spec.models, spec.repeats,
                                       spec.max_cost_usd, spec.label, saved, total)
        self._launch(run_id, target, tasks, spec, saved)
        return run_id

    def resume(self, run_id: int, max_cost_usd: float | None = None) -> None:
        """Carry on a run that stopped, skipping what's already done."""
        row = self.store.run(run_id)
        if row is None:
            raise RunRefused("no such run")
        if self.live_state(row) == "running":
            raise RunRefused("that run is still going", conflict=True)
        spec = RunSpec(row["target"], row["suite"], row["models"], row["repeats"],
                       row["max_cost_usd"] if max_cost_usd is None else max_cost_usd, row["label"])
        target, tasks = self._load(spec)
        self._guard(target.agent_id)
        saved = self.store.selection(target.agent_id)
        self.store.update_run(run_id, state="running", max_cost_usd=spec.max_cost_usd, saved_selection=saved,
                              error=None, ended_at=None)
        self._launch(run_id, target, tasks, spec, saved)

    def live_state(self, row: dict) -> str:
        """A row can say running with nothing behind it, if a database failure stopped it being updated."""
        task = self._tasks.get(row["id"])
        return "interrupted" if row["state"] == "running" and (task is None or task.done()) else row["state"]

    def just_ended(self, agent_id: str) -> bool:
        """After a run the agent may still be mid-turn. Its late calls are refused: no spending, no live tools."""
        return self._ended.get(agent_id, 0.0) > time.monotonic()

    def stop(self, run_id: int) -> None:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            raise RunRefused("that run isn't going", conflict=True)
        self._stopping.add(run_id)
        task.cancel()

    async def wait(self, run_id: int) -> None:
        if task := self._tasks.get(run_id):
            await asyncio.gather(task, return_exceptions=True)

    async def aclose(self) -> None:
        """Shutting down: anything still going is interrupted, and cleans up after itself."""
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def recover(self) -> None:
        """At startup: runs left going by a crash are interrupted, and their agents get their model back."""
        for row in self.store.runs(state="running"):
            self._ended[row["agent_id"]] = time.monotonic() + DEFAULT_TIMEOUT_S
            self.store.select(row["agent_id"], row["saved_selection"], "run")
            self.store.update_run(row["id"], state="interrupted", error="ProvingBench stopped during this run",
                                  ended_at=self.now().isoformat())

    def estimate(self, spec: RunSpec) -> float | None:
        """A rough cost before starting; None if a model's price isn't known."""
        _, tasks = self._load(spec)
        attempts = sum(self._repeats(task, spec) for task in tasks)
        total = 0.0
        for model in spec.models:
            if not is_paid(self.cfg, self.store, model):
                continue
            row = self.store.model(model)
            if not row or row["price_prompt"] is None or row["price_completion"] is None:
                return None
            total += attempts * EST_CALLS * (EST_PROMPT_TOKENS * row["price_prompt"]
                                             + EST_COMPLETION_TOKENS * row["price_completion"])
        return total

    def _ended_state(self, run_id: int) -> str:
        return "stopped" if run_id in self._stopping else "interrupted"

    def _finish(self, run_id: int, agent_id: str, saved: str | None, state: str, error: str | None) -> None:
        """Give the agent its model back and record how the run ended. Only the first call does anything."""
        if agent_id not in self.active or self.active[agent_id].run_id != run_id:
            return
        active = self.active.pop(agent_id)
        active.shared.closed = True
        self._ended[agent_id] = time.monotonic() + active.grace_s
        self._stopping.discard(run_id)
        # One failing must not skip the other. If both fail, live_state and the next start's recover() cover it.
        for step in (lambda: self.store.select(agent_id, saved, "run"),
                     lambda: self.store.update_run(run_id, state=state, error=error, ended_at=self.now().isoformat())):
            try:
                step()
            except Exception:
                log.exception("couldn't finish recording run %s", run_id)

    async def _run(self, run_id: int, target: Target, tasks: list[Task], spec: RunSpec, saved: str | None) -> None:
        state, error = "done", None
        active = self.active[target.agent_id]
        client = httpx.AsyncClient(transport=self._transport, trust_env=False)
        done = self.store.done_attempts(run_id)
        try:
            for model in spec.models:
                active.model = model
                self.store.select(target.agent_id, model, "run")
                for task in tasks:
                    for repeat in range(self._repeats(task, spec)):
                        if (model, task.id, repeat) in done:
                            continue
                        if self.store.spend(run_id) >= spec.max_cost_usd:
                            state = "stopped_cost"
                            return
                        error = await self._attempt(run_id, target, client, active, model, task, repeat)
                        if active.shared.over_limit:
                            state, error = "stopped_cost", active.shared.over_limit
                            return
                        if error:
                            state = "failed"
                            return
        except asyncio.CancelledError:
            state = self._ended_state(run_id)
            raise
        except Exception as e:
            state, error = "failed", f"{type(e).__name__}: {e}"
            log.exception("run %s failed", run_id)
        finally:
            self._finish(run_id, target.agent_id, saved, state, error)
            with anyio.CancelScope(shield=True):
                await client.aclose()

    async def _attempt(self, run_id, target, client, active, model, task: Task, repeat: int) -> str | None:
        """Run and grade one attempt. Returns an error only if the run can't go on: nothing is saved for it."""
        attempt_id = uuid.uuid4().hex
        active.task_id, active.attempt_id, active.stubs, active.allow = task.id, attempt_id, task.stubs, task.allow
        session = f"tm-{secrets.token_hex(4)}"
        log_errors = self.store.log_errors
        started, reply, error = time.monotonic(), None, None
        try:
            for turn in task.turns:
                answer = await send(target, client, session, turn)
                if answer.unreachable:
                    return answer.error
                if answer.error:
                    error = answer.error
                    break
                reply = answer.reply
        finally:
            active.task_id, active.attempt_id, active.stubs, active.allow = None, None, {}, []
        if active.shared.over_limit:
            return None     # cut short by the limit, so it's no measure of the model; resuming runs it again
        seconds = time.monotonic() - started
        calls = self.store.attempt_calls(attempt_id)
        if self.store.log_errors > log_errors:
            outcome, results, error = "error", [], LOG_FAILED
        elif error is None and not calls:
            return NO_CALLS.format(agent_id=target.agent_id)
        else:
            outcome, results = await grade(task, Attempt(calls, reply, error, seconds, task.turns, run_id), self.judge)
        self.store.add_result(run_id, model, task.id, task.hash, task.tags, repeat, attempt_id, outcome,
                              [r.to_json() for r in results], reply, error, seconds,
                              sum(c["cost_usd"] or 0 for c in calls))
        return None
