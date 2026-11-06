"""Task suites: a folder of TOML files we write ourselves. A mistake in one stops the run and names the file."""
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CHECK_TYPES = ("tool_called", "tool_not_called", "no_unknown_tools", "reply_not_empty", "reply_matches",
               "reply_not_matches", "no_promise_without_action", "max_tool_rounds", "max_seconds", "judge")


class SuiteError(Exception):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    tags: list[str]
    turns: list[str]
    stubs: dict[str, str]
    allow: list[str]
    checks: list[dict]
    repeats: int | None
    hash: str


def _is_pattern(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        re.compile(value)
    except re.error:
        return False
    return True


def _check_problem(check) -> str | None:
    kind = check.get("type") if isinstance(check, dict) else None
    if kind not in CHECK_TYPES:
        return f"unknown check type {kind!r}"
    if kind in ("tool_called", "tool_not_called") and not isinstance(check.get("name"), str):
        return f"{kind} needs a name"
    if kind == "tool_called" and "arguments" in check:
        arguments = check["arguments"]
        if not isinstance(arguments, dict) or not all(map(_is_pattern, arguments.values())):
            return "tool_called arguments must map each key to a valid regular expression"
    if kind in ("reply_matches", "reply_not_matches") and not _is_pattern(check.get("pattern")):
        return f"{kind} needs a valid regular expression as its pattern"
    if kind in ("max_tool_rounds", "max_seconds") and (
            not isinstance(check.get("limit"), int | float) or isinstance(check["limit"], bool)):
        return f"{kind} needs a number as its limit"
    if kind == "judge" and not (isinstance(check.get("rubric"), str) and check["rubric"].strip()):
        return "judge needs a rubric"
    return None


def _task(data: dict, digest: str) -> Task:
    for key in ("id", "title"):
        if not (isinstance(data.get(key), str) and data[key].strip()):
            raise SuiteError(f"{key} must be some text")
    turns = data.get("turns")
    if not (isinstance(turns, list) and turns):
        raise SuiteError("needs at least one [[turns]] entry")
    if not all(isinstance(t, dict) and isinstance(t.get("say"), str) and t["say"].strip() for t in turns):
        raise SuiteError("each [[turns]] entry needs some text in say")
    checks = data.get("checks")
    if not (isinstance(checks, list) and checks):
        raise SuiteError("needs at least one [[checks]] entry")
    for check in checks:
        if problem := _check_problem(check):
            raise SuiteError(problem)
    stubs = data.get("stubs", {})
    if not (isinstance(stubs, dict) and all(isinstance(v, str) for v in stubs.values())):
        raise SuiteError("each entry in [stubs] must be some text")
    policy = data.get("policy", {})
    allow = policy.get("allow", []) if isinstance(policy, dict) else None
    if not (isinstance(allow, list) and all(isinstance(a, str) for a in allow)):
        raise SuiteError("policy.allow must be a list of tool names")
    if both := set(stubs) & set(allow):
        raise SuiteError(f"{', '.join(sorted(both))} is both stubbed and allowed; pick one")
    repeats = data.get("repeats")
    if repeats is not None and (not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1):
        raise SuiteError("repeats must be a whole number of at least 1")
    tags = data.get("tags", [])
    if not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
        raise SuiteError("tags must be a list of words")
    return Task(data["id"], data["title"], tags, [t["say"] for t in turns], stubs, allow, checks, repeats, digest)


def load_suite(path: Path) -> list[Task]:
    files = sorted(path.glob("*.toml")) if path.is_dir() else []
    if not files:
        raise SuiteError(f"{path}: no task files found")
    tasks: dict[str, Task] = {}
    for file in files:
        try:
            raw = file.read_bytes()
            task = _task(tomllib.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, SuiteError) as e:
            raise SuiteError(f"{file}: {e}") from e
        if task.id in tasks:
            raise SuiteError(f"{file}: the id {task.id!r} is already used by another task")
        tasks[task.id] = task
    return list(tasks.values())
