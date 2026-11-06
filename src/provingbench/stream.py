"""Reading and writing OpenAI-style answers. The relay forwards a stream's bytes as they are and keeps a parsed copy."""
import json
import re
import time
from dataclasses import dataclass, field

_EVENT_END = re.compile(rb"\r?\n\r?\n")


@dataclass
class Completion:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish: str | None = None
    usage: dict | None = None
    model: str | None = None
    raw_tokens: str = ""        # the logprob tokens, which can show what a server threw away


class Relay:
    """Feed it upstream bytes; it returns what to forward. Whatever arrives, it never raises."""

    def __init__(self, drop_usage: bool):
        self.drop_usage = drop_usage
        self.started = False
        self._buffer = b""
        self._text: list[str] = []
        self._raw: list[str] = []
        self._calls: dict[int, dict] = {}
        self._finish = self._usage = self._model = None

    def feed(self, chunk: bytes) -> bytes:
        self._buffer += chunk
        out = []
        while match := _EVENT_END.search(self._buffer):
            event, self._buffer = self._buffer[:match.end()], self._buffer[match.end():]
            if self._read(event[:match.start()]):
                out.append(event)
        return b"".join(out)

    def flush(self) -> bytes:
        rest, self._buffer = self._buffer, b""
        return rest

    def _read(self, event: bytes) -> bool:
        """Parse one event; False means it isn't to be forwarded."""
        for line in event.splitlines():
            if not line.startswith(b"data:"):
                continue
            try:
                data = json.loads(line[5:])
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            choices = data.get("choices")
            if isinstance(data.get("usage"), dict):
                self._usage = data["usage"]
                if self.drop_usage and not choices:
                    return False
            if isinstance(data.get("model"), str):
                self._model = data["model"]
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                self._choice(choices[0])
        return True

    def _choice(self, choice: dict) -> None:
        if isinstance(choice.get("finish_reason"), str):
            self._finish = choice["finish_reason"]
        tokens = _logprob_tokens(choice)
        if tokens:
            self._raw.append(tokens)
            self.started = True
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return
        if isinstance(delta.get("content"), str) and delta["content"]:
            self._text.append(delta["content"])
            self.started = True
        if isinstance(delta.get("tool_calls"), list) and delta["tool_calls"]:
            _merge(self._calls, delta["tool_calls"])
            self.started = True

    def completion(self) -> Completion:
        return Completion("".join(self._text), _assembled(self._calls), self._finish, self._usage, self._model,
                          "".join(self._raw))


def _logprob_tokens(choice: dict) -> str:
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        return ""
    return "".join(t["token"] for t in content if isinstance(t, dict) and isinstance(t.get("token"), str))


def _merge(calls: dict[int, dict], parts: list) -> None:
    for part in parts:
        if not isinstance(part, dict):
            continue
        index = part.get("index", 0)
        call = calls.setdefault(index if isinstance(index, int) else 0, {"id": "", "name": "", "arguments": ""})
        fn = part.get("function") if isinstance(part.get("function"), dict) else {}
        call["id"] = call["id"] or (part.get("id") if isinstance(part.get("id"), str) else "")
        call["name"] = call["name"] or (fn.get("name") if isinstance(fn.get("name"), str) else "")
        if isinstance(fn.get("arguments"), str):
            call["arguments"] += fn["arguments"]


def _assembled(calls: dict[int, dict]) -> list[dict]:
    return [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for _, c in sorted(calls.items()) if c["name"]]


def from_json(body: dict) -> Completion:
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    merged: dict[int, dict] = {}
    _merge(merged, [{**c, "index": i} for i, c in enumerate(calls) if isinstance(c, dict)])
    text = message.get("content")
    return Completion(text if isinstance(text, str) else "", _assembled(merged), choice.get("finish_reason"),
                      body.get("usage") if isinstance(body.get("usage"), dict) else None, body.get("model"),
                      _logprob_tokens(choice))


def _indexed(c: Completion) -> list[dict]:
    return [{**call, "index": i} for i, call in enumerate(c.tool_calls)]


def _envelope(model: str, kind: str, choice: dict) -> dict:
    return {"id": "chatcmpl-provingbench", "object": kind, "created": int(time.time()), "model": model, "choices": [choice]}


def render_sse(c: Completion, model: str) -> bytes:
    """A buffered answer as one short stream, for an agent that asked to stream."""
    finish = c.finish or ("tool_calls" if c.tool_calls else "stop")
    delta: dict = {"role": "assistant", "content": c.text}
    if c.tool_calls:
        delta["tool_calls"] = _indexed(c)
    chunks = [_envelope(model, "chat.completion.chunk", {"index": 0, "delta": delta, "finish_reason": None}),
              _envelope(model, "chat.completion.chunk", {"index": 0, "delta": {}, "finish_reason": finish})]
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"


def render_json(c: Completion, model: str) -> dict:
    message: dict = {"role": "assistant", "content": c.text}
    if c.tool_calls:
        message["tool_calls"] = c.tool_calls
    body = _envelope(model, "chat.completion", {"index": 0, "message": message,
                                                "finish_reason": c.finish or ("tool_calls" if c.tool_calls else "stop")})
    if c.usage:
        body["usage"] = c.usage
    return body
