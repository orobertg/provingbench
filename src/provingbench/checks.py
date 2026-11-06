"""Grading one attempt. Every result says why, so a grade can always be explained."""
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from provingbench.flags import promises_action
from provingbench.suite import Task

# Flags that are the model's doing, as opposed to trouble reaching it.
MODEL_FLAGS = ("empty_reply", "dropped_tool_call", "unknown_tool", "bad_arguments", "rounds_exhausted")


class JudgeFailed(Exception):
    pass


@dataclass(frozen=True)
class Attempt:
    calls: list[dict]
    reply: str | None
    error: str | None
    seconds: float
    turns: list[str]
    run_id: int | None = None


@dataclass(frozen=True)
class CheckResult:
    type: str
    status: str     # pass | fail | skipped
    reason: str

    def to_json(self) -> dict:
        return asdict(self)


def _flags(attempt: Attempt) -> list[dict]:
    return [f for call in attempt.calls for f in call["flags"]]


def _tools(attempt: Attempt) -> list[dict]:
    return [t for call in attempt.calls for t in call["tool_calls"]]


def _arguments(tool: dict) -> dict:
    try:
        args = json.loads(tool["arguments"] or "{}")
    except ValueError:
        return {}
    return args if isinstance(args, dict) else {}


def _tool_called(check: dict, attempt: Attempt) -> tuple[bool, str]:
    name, wanted = check["name"], check.get("arguments", {})
    same = [t for t in _tools(attempt) if t["name"] == name]
    if not same:
        return False, f"{name} was not called"
    for tool in same:
        args = _arguments(tool)
        if all(key in args and re.search(pattern, str(args[key])) for key, pattern in wanted.items()):
            return True, f"{name} was called"
    return False, f"{name} was called, but not with arguments matching {wanted}: {[t['arguments'] for t in same]}"


def _tool_not_called(check: dict, attempt: Attempt) -> tuple[bool, str]:
    called = any(t["name"] == check["name"] for t in _tools(attempt))
    return not called, f"{check['name']} was {'called' if called else 'not called'}"


def _no_unknown_tools(_, attempt: Attempt) -> tuple[bool, str]:
    bad = [f["detail"] or f["name"] for f in _flags(attempt) if f["name"] in ("unknown_tool", "dropped_tool_call")]
    return not bad, f"called tools that weren't offered: {', '.join(bad)}" if bad else "called only offered tools"


def _reply_not_empty(_, attempt: Attempt) -> tuple[bool, str]:
    empty = not (attempt.reply or "").strip()
    return not empty, "the reply was empty" if empty else "there was a reply"


def _reply_matches(check: dict, attempt: Attempt) -> tuple[bool, str]:
    found = re.search(check["pattern"], attempt.reply or "") is not None
    return found, f"the reply {'matches' if found else 'does not match'} {check['pattern']}"


def _reply_not_matches(check: dict, attempt: Attempt) -> tuple[bool, str]:
    found = re.search(check["pattern"], attempt.reply or "") is not None
    return not found, f"the reply {'matches' if found else 'does not match'} {check['pattern']}"


def _no_promise(_, attempt: Attempt) -> tuple[bool, str]:
    promised = promises_action(attempt.reply or "") and not _tools(attempt)
    return not promised, "promised to act but called no tool" if promised else "no empty promise"


def _max_tool_rounds(check: dict, attempt: Attempt) -> tuple[bool, str]:
    rounds = sum(1 for call in attempt.calls if call["tool_calls"])
    return rounds <= check["limit"], f"{rounds} tool rounds, limit {check['limit']}"


def _max_seconds(check: dict, attempt: Attempt) -> tuple[bool, str]:
    return attempt.seconds <= check["limit"], f"took {attempt.seconds:.1f} s, limit {check['limit']}"


_CHECKS: dict[str, Callable] = {
    "tool_called": _tool_called, "tool_not_called": _tool_not_called, "no_unknown_tools": _no_unknown_tools,
    "reply_not_empty": _reply_not_empty, "reply_matches": _reply_matches, "reply_not_matches": _reply_not_matches,
    "no_promise_without_action": _no_promise, "max_tool_rounds": _max_tool_rounds, "max_seconds": _max_seconds}


async def _judged(check: dict, attempt: Attempt, judge) -> CheckResult:
    if judge is None:
        return CheckResult("judge", "skipped", "no judge model is configured")
    try:
        passed, reason = await judge(check["rubric"], attempt)
    except JudgeFailed as e:
        return CheckResult("judge", "skipped", f"the judge couldn't decide: {e}")
    model = getattr(judge, "model", None)
    return CheckResult("judge", "pass" if passed else "fail", f"{reason} (judged by {model})" if model else reason)


async def grade(task: Task, attempt: Attempt,
                judge: Callable[[str, Attempt], Awaitable[tuple[bool, str]]] | None = None) -> tuple[str, list[CheckResult]]:
    """(outcome, results). The outcome is pass, fail, or error when the attempt couldn't be graded."""
    names = [f["name"] for f in _flags(attempt)]
    if attempt.error:
        blamed = next((n for n in MODEL_FLAGS if n in names), None)
        if blamed is None:
            return "error", []
        return "fail", [CheckResult("model_failure", "fail", f"{blamed}, and the agent answered: {attempt.error}")]
    results = []
    for check in task.checks:
        if check["type"] == "judge":
            results.append(await _judged(check, attempt, judge))
        else:
            passed, reason = _CHECKS[check["type"]](check, attempt)
            results.append(CheckResult(check["type"], "pass" if passed else "fail", reason))
    if "rounds_exhausted" in names:
        results.append(CheckResult("model_failure", "fail", "kept calling tools for all 8 rounds"))
    if any(r.status == "fail" for r in results):
        return "fail", results
    # Every check skipped (the judge was out, say) means nothing was measured.
    return ("pass" if any(r.status == "pass" for r in results) else "error"), results
