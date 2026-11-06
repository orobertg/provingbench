"""Stand-ins: a scripted model server, and a small agent that talks to the proxy."""
import asyncio
import json

import httpx

from provingbench.stream import Relay


def event(data: dict) -> bytes:
    return b"data: " + json.dumps(data).encode() + b"\n\n"


def sse(text: str = "", usage: dict | None = None) -> httpx.Response:
    body = event({"model": "up", "choices": [{"delta": {"content": text}}]})
    body += event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    if usage:
        body += event({"choices": [], "usage": usage})
    return httpx.Response(200, content=body + b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"})


def sse_calls(*calls: tuple[str, str]) -> httpx.Response:
    parts = [{"index": i, "id": f"c{i + 1}", "type": "function", "function": {"name": name, "arguments": args}}
             for i, (name, args) in enumerate(calls)]
    body = event({"choices": [{"delta": {"tool_calls": parts}}]})
    body += event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    return httpx.Response(200, content=body + b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"})


def sse_call(name: str, args: str = "{}") -> httpx.Response:
    return sse_calls((name, args))


def json_reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"model": "up", "choices": [
        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


class _Endless(httpx.AsyncByteStream):
    def __init__(self, owner, first: bytes):
        self.owner, self.first = owner, first

    async def __aiter__(self):
        yield self.first
        await asyncio.Event().wait()

    async def aclose(self):
        self.owner.closed = True


class Breaks(httpx.AsyncByteStream):
    """Some text, then the connection dies."""

    async def __aiter__(self):
        yield event({"choices": [{"delta": {"content": "par"}}]})
        raise httpx.ReadError("gone")


class FakeUpstream:
    """A scripted model server: `script` queues answers (a Response, or an exception to raise)."""

    def __init__(self):
        self.answers, self.requests, self.bodies = [], [], []
        self.forever = None
        self.closed = False
        self.gate: asyncio.Event | None = None

    def script(self, *answers) -> None:
        self.answers.extend(answers)

    def script_forever(self, answer) -> None:
        self.forever = answer

    def endless(self, text: str) -> None:
        self.script(httpx.Response(200, stream=_Endless(self, event({"choices": [{"delta": {"content": text}}]})),
                                   headers={"content-type": "text/event-stream"}))

    def hold(self) -> None:
        """Answers wait until `release`."""
        self.gate = asyncio.Event()

    def release(self) -> None:
        self.gate.set()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        if self.gate:
            await self.gate.wait()
        answer = self.answers.pop(0) if self.answers else self.forever
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            raise AssertionError("the fake upstream has no answer scripted")
        if isinstance(answer.stream, _Endless | Breaks):
            return answer
        return httpx.Response(answer.status_code, headers=answer.headers, content=answer.content)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


class FakeAgent:
    """Takes {session_id, text}, asks the proxy in a loop, runs whatever tool calls reach it, returns {reply}."""
    TOOLS = [{"type": "function", "function": {"name": "system_report", "parameters": {"type": "object"}}}]

    def __init__(self):
        self.proxy = None   # set once the app exists
        self.sessions, self.ran_tools = [], []
        self.crash_on_turn = self.down = self.skips_proxy = False
        self.pause_on_turn = None       # this turn waits forever, with `paused` set
        self.paused = asyncio.Event()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused")
        body = json.loads(request.content)
        self.sessions.append(body["session_id"])
        if len(self.sessions) == self.pause_on_turn:
            self.paused.set()
            await asyncio.Event().wait()
        if self.crash_on_turn:
            return httpx.Response(500, json={"error": "crashed"})
        if self.skips_proxy:
            return httpx.Response(200, json={"reply": "answered on my own"})
        messages = [{"role": "user", "content": body["text"]}]
        for _ in range(10):
            resp = await self.proxy.post("/a/bot/v1/chat/completions", json={
                "model": "whatever", "stream": True, "tools": self.TOOLS, "messages": messages})
            if resp.status_code >= 400:
                return httpx.Response(502, json={"error": resp.text})
            relay = Relay(drop_usage=False)
            relay.feed(resp.content)
            c = relay.completion()
            if not c.tool_calls:
                return httpx.Response(200, json={"reply": c.text})
            self.ran_tools += [call["function"]["name"] for call in c.tool_calls]
            messages.append({"role": "assistant", "content": c.text, "tool_calls": c.tool_calls})
            messages += [{"role": "tool", "tool_call_id": call["id"], "content": "real result"} for call in c.tool_calls]
        return httpx.Response(500, json={"error": "too many rounds"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)
