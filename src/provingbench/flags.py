"""What can go wrong with one model call, found by looking at the request and the answer."""
import json
import re

from provingbench.store import Flag
from provingbench.stream import Completion

_PROMISE = re.compile(
    r"\b(let me|i'll|i will|i'm going to|i am going to|allow me to)\s+(just\s+|quickly\s+|now\s+)?"
    r"(check|look|see|find|verify|run|test|get|fetch|search|inspect|confirm|review|try|start|restart)\b"
    r"|\b(one moment|give me a (moment|second|sec)|hold on|just a (moment|second|sec))\b"
    r"|(^|[.!?]\s+)checking\b", re.I)
_DROPPED_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')
_TYPES = {"string": str, "boolean": bool, "array": list, "object": dict}


def promises_action(text: str) -> bool:
    return _PROMISE.search(text.strip()) is not None


def _offered(request: dict) -> dict[str, dict]:
    tools = request.get("tools")
    found = {}
    for tool in tools if isinstance(tools, list) else ():
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            found[fn["name"]] = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    return found


def _fits(value, kind) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return not isinstance(kind, str) or kind not in _TYPES or isinstance(value, _TYPES[kind])


def _arguments_problem(raw: str, schema: dict | None) -> str | None:
    try:
        args = json.loads(raw or "{}")
    except ValueError:
        return "not valid JSON"
    if not isinstance(args, dict):
        return "not a JSON object"
    if schema is None:
        return None
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    if missing := [key for key in required if key not in args]:
        return f"missing {', '.join(map(str, missing))}"
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for key, value in args.items():
        prop = props.get(key)
        if isinstance(prop, dict) and not _fits(value, prop.get("type")):
            return f"{key} should be {prop.get('type')}"
    return None


def detect(request: dict, c: Completion, *, ttft_s: float | None, slow_s: float) -> list[Flag]:
    flags: list[Flag] = []
    offered = _offered(request)
    for call in c.tool_calls:
        fn = call["function"]
        if fn["name"] not in offered:
            flags.append(Flag("unknown_tool", fn["name"]))
        if problem := _arguments_problem(fn["arguments"], offered.get(fn["name"])):
            flags.append(Flag("bad_arguments", f"{fn['name']}: {problem}"))
    if not c.text.strip() and not c.tool_calls:
        flags.append(Flag("empty_reply"))
        match = _DROPPED_NAME.search(c.raw_tokens)
        if match or "<tool_call>" in c.raw_tokens:
            flags.append(Flag("dropped_tool_call", match.group(1) if match else ""))
    elif not c.tool_calls and offered and promises_action(c.text):
        flags.append(Flag("promised_no_action"))
    if ttft_s is not None and ttft_s > slow_s:
        flags.append(Flag("slow", f"{ttft_s:.1f} s to first token"))
    return flags
