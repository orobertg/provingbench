"""During a test run the model's tool calls stop here: only calls a task allows ever reach the agent."""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from provingbench.store import Flag
from provingbench.stream import Completion

BLOCKED = "blocked by the evaluator"
MAX_ROUNDS = 8
OUT_OF_ROUNDS = f"(the evaluator stopped this after {MAX_ROUNDS} tool rounds)"


@dataclass
class RunStatus:
    """Shared by a run and every request in flight for it."""
    closed: bool = False
    over_limit: str | None = None


@dataclass
class ActiveRun:
    """A run in progress for one agent. The runner sets the task's fields per attempt and clears them in between.

    Each request takes a copy when it arrives, so its rounds stay under the attempt they began in."""
    run_id: int
    task_id: str | None = None
    attempt_id: str | None = None
    stubs: dict[str, str] = field(default_factory=dict)
    allow: list[str] = field(default_factory=list)
    model: str | None = None
    limit_usd: float = float("inf")
    grace_s: float = 300.0
    shared: RunStatus = field(default_factory=RunStatus)


def _rows(c: Completion, active: ActiveRun) -> list[dict]:
    return [{"id": call["id"], "name": call["function"]["name"], "arguments": call["function"]["arguments"],
             "answered": "stubbed" if call["function"]["name"] in active.stubs else "blocked"}
            for call in c.tool_calls]


def _with_answers(body: dict, c: Completion, active: ActiveRun) -> dict:
    """The conversation so far, plus the model's calls and the answers the proxy gives them."""
    answers = [{"role": "tool", "tool_call_id": call["id"],
                "content": active.stubs.get(call["function"]["name"], BLOCKED)} for call in c.tool_calls]
    said = {"role": "assistant", "content": c.text, "tool_calls": c.tool_calls}
    return {**body, "messages": [*body["messages"], said, *answers]}


async def answer(body: dict, active: ActiveRun, ask: Callable[[dict], Awaitable[Completion]],
                 log: Callable[..., None]) -> Completion:
    """What the agent should get back. `log(body, c, rows, flags)` records each round; rows None means as they came."""
    for round_number in range(1, MAX_ROUNDS + 1):
        c = await ask(body)
        if all(call["function"]["name"] in active.allow for call in c.tool_calls):
            log(body, c, None, ())
            return c
        last = round_number == MAX_ROUNDS
        log(body, c, _rows(c, active), (Flag("rounds_exhausted"),) if last else ())
        body = _with_answers(body, c, active)
    return Completion(text=OUT_OF_ROUNDS, finish="stop")
