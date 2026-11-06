"""The model endpoint agents point at. Live calls pass straight through while a copy goes to the log."""
import asyncio
import copy
import json
import logging
import re
import time
from dataclasses import dataclass, replace

import anyio
import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from provingbench.flags import detect
from provingbench.intercept import answer
from provingbench.store import CallRecord, Flag
from provingbench.stream import Completion, Relay, from_json, render_json, render_sse
from provingbench.upstreams import cost_of, is_paid, route

log = logging.getLogger("provingbench")
_AGENT_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_PATH = "/chat/completions"
_PASSED_HEADERS = ("content-type", "retry-after")
KEEPALIVE_S = 15.0
_DISCONNECT_POLL_S = 1.0
_STATUS_CLIENT_GONE = 499


def valid_agent(agent: str) -> bool:
    return _AGENT_ID.fullmatch(agent) is not None


def error_body(message: str, code: str = "proxy_error") -> dict:
    return {"error": {"message": message, "type": code, "code": code}}


def safely(store, action, *args, default=None, **kw):
    """Run a store action; if the database fails, count it and carry on, because the call must still go through."""
    try:
        return action(*args, **kw)
    except Exception:
        store.log_errors += 1
        log.warning("couldn't use the log database", exc_info=True)
        return default


@dataclass(frozen=True)
class Call:
    """One agent's request, resolved to where it's going."""
    agent: str
    request: dict       # as the agent sent it
    forward: dict       # as the upstream gets it
    upstream: object
    model: str          # the full model ID, such as ollama/qwen3
    drop_usage: bool
    started: float
    run: object = None  # this request's copy of the ActiveRun, during a test run
    selected: bool = False  # the owner or the run chose the model, rather than the request naming it


def forward_body(cfg, body: dict, upstream, name: str) -> dict:
    out = {**body, "model": name}
    if body.get("stream"):
        out["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    if upstream.name == "openrouter":
        out["usage"] = {"include": True}
    if upstream.name == "ollama" and cfg.ollama_logprobs:
        out["logprobs"] = True
    return out


def tool_rows(c: Completion, answered: str = "real") -> list[dict]:
    return [{"id": call["id"], "name": call["function"]["name"], "arguments": call["function"]["arguments"],
             "answered": answered} for call in c.tool_calls]


def save_call(app, call: Call, c: Completion, *, state: str, status: int | None, ttft_s: float | None,
              extra_flags: tuple[Flag, ...] = (), rows: list[dict] | None = None, error: str = "") -> None:
    """Log one finished call. Nothing that goes wrong here reaches the agent."""
    store = app.state.store

    def write():
        flags = detect(call.request, c, ttft_s=ttft_s, slow_s=app.state.cfg.slow_first_token_s) if state == "ok" else []
        response = {"text": c.text, "model": c.model}
        if c.raw_tokens:
            response["raw_tokens"] = c.raw_tokens
        if error:
            response["error"] = error
        usage = c.usage or {}
        store.add_call(CallRecord(
            agent_id=call.agent, model=call.model, upstream=call.upstream.name, state=state, status=status,
            request=call.request, response=response, tool_calls=tool_rows(c) if rows is None else rows, finish=c.finish,
            ttft_s=ttft_s, total_s=time.monotonic() - call.started, prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=cost_of(app.state.cfg, store, call.model, c.usage), flags=[*flags, *extra_flags],
            run_id=call.run.run_id if call.run else None, task_id=call.run.task_id if call.run else None,
            attempt_id=call.run.attempt_id if call.run else None))
    safely(store, write)


def resolve(app, agent: str, body: dict) -> Call:
    cfg, store = app.state.cfg, app.state.store
    active = app.state.active.get(agent)
    selected = (active and active.model) or safely(store, store.selection, agent)
    upstream, name = route(cfg, selected or str(body.get("model") or ""))
    wants_usage = isinstance(body.get("stream_options"), dict) and body["stream_options"].get("include_usage")
    return Call(agent, body, forward_body(cfg, body, upstream, name), upstream, f"{upstream.name}/{name}",
                not wants_usage, time.monotonic(), copy.copy(active) if active else None, bool(selected))


def _refused_paid(app, call: Call, request: Request) -> bool:
    """A paid model named in the request itself needs to be on the agent's list. Unknown in a locked database: refuse."""
    if call.selected or request.state.role == "owner":
        return False
    store, cfg = app.state.store, app.state.cfg
    return safely(store, lambda: is_paid(cfg, store, call.model) and call.model not in store.allowed_models(call.agent),
                  default=True)


def _failed(app, call: Call, e: Exception) -> JSONResponse:
    message = f"couldn't reach {call.upstream.name}: {e}"
    save_call(app, call, Completion(), state="error", status=502, ttft_s=None, error=message,
              extra_flags=(Flag("upstream_error", message),))
    return JSONResponse(error_body(message), status_code=502)


def _passed_on(app, call: Call, status: int, content: bytes, headers) -> Response:
    """An upstream's error answer, unchanged."""
    text = content.decode(errors="replace")
    save_call(app, call, Completion(), state="error", status=status, ttft_s=None, error=text[:2000],
              extra_flags=(Flag("upstream_error", f"HTTP {status}"),))
    kept = {k: headers[k] for k in _PASSED_HEADERS if k in headers}
    return Response(content, status_code=status, headers=kept)


async def _relay(app, call: Call, resp: httpx.Response):
    relay = Relay(drop_usage=call.drop_usage)
    state, ttft_s, extra, error = "ok", None, (), ""
    try:
        async for chunk in resp.aiter_bytes():
            out = relay.feed(chunk)
            if ttft_s is None and relay.started:
                ttft_s = time.monotonic() - call.started
            if out:
                yield out
        if tail := relay.flush():
            yield tail
    except httpx.HTTPError as e:
        state, error = "error", f"stream broke: {e}"
        extra = (Flag("upstream_error", error),)
        raise       # the agent must see a broken stream, not a finished one
    except (asyncio.CancelledError, GeneratorExit):
        state = "cancelled"
        raise
    finally:
        with anyio.CancelScope(shield=True):
            await resp.aclose()
        save_call(app, call, relay.completion(), state=state, status=resp.status_code, ttft_s=ttft_s,
                  extra_flags=extra, error=error)


async def _live_stream(app, call: Call) -> Response:
    client = app.state.ups.client(call.upstream)
    try:
        resp = await client.send(client.build_request("POST", _PATH, json=call.forward), stream=True)
    except httpx.HTTPError as e:
        return _failed(app, call, e)
    if resp.status_code >= 400:
        try:
            content = await resp.aread()
        finally:
            await resp.aclose()
        return _passed_on(app, call, resp.status_code, content, resp.headers)
    return StreamingResponse(_relay(app, call, resp), status_code=resp.status_code,
                             media_type=resp.headers.get("content-type", "text/event-stream"))


async def _live_plain(app, call: Call) -> Response:
    client = app.state.ups.client(call.upstream)
    try:
        resp = await client.post(_PATH, json=call.forward)
    except httpx.HTTPError as e:
        return _failed(app, call, e)
    if resp.status_code >= 400:
        return _passed_on(app, call, resp.status_code, resp.content, resp.headers)
    try:
        c = from_json(resp.json())
    except (ValueError, AttributeError):
        c = Completion()
    save_call(app, call, c, state="ok", status=resp.status_code, ttft_s=None)
    return Response(resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


class _RoundFailed(Exception):
    def __init__(self, response: Response):
        self.response = response


def _check_run_may_spend(app, call: Call) -> None:
    run, store = call.run, app.state.store
    if run.shared.closed:
        raise _RoundFailed(JSONResponse(error_body("this test run has ended", "run_ended"), status_code=409))
    if not run.shared.over_limit and safely(store, store.spend, run.run_id, default=0.0) >= run.limit_usd:
        run.shared.over_limit = f"reached the run's limit of ${run.limit_usd:.2f}"
    if run.shared.over_limit:
        raise _RoundFailed(JSONResponse(error_body(f"this test run stopped: {run.shared.over_limit}", "over_limit"),
                                        status_code=402))


def _note_unknown_cost(app, call: Call, c: Completion) -> None:
    """A paid call whose cost can't be told would blind the limit, so the run stops there."""
    cfg, store = app.state.cfg, app.state.store
    if safely(store, lambda: is_paid(cfg, store, call.model) and cost_of(cfg, store, call.model, c.usage) is None):
        call.run.shared.over_limit = f"couldn't tell what a call to {call.model} cost, so the limit can't be kept"


def _finished(call: Call, c: Completion) -> Response:
    if call.request.get("stream"):
        return Response(render_sse(c, call.model), media_type="text/event-stream")
    return JSONResponse(render_json(c, call.model))


async def _kept_alive(task: asyncio.Task, call: Call):
    """Rounds still going: comments keep a streaming agent from timing out. The status is already sent, so a failure
    from here on is an error event."""
    try:
        while not task.done():
            yield b": keep-alive\n\n"
            await asyncio.wait({task}, timeout=KEEPALIVE_S)
        try:
            yield render_sse(task.result(), call.model)
        except _RoundFailed as e:
            yield b"data: " + json.dumps(error_body(e.response.body.decode(errors="replace")[:500])).encode() + b"\n\n"
    finally:
        task.cancel()


async def _ask_round(app, call: Call, body: dict, timings: dict) -> Completion:
    """One buffered model call of a test run, streamed from upstream so timing and logprobs match live use."""
    _check_run_may_spend(app, call)
    client = app.state.ups.client(call.upstream)
    started = time.monotonic()
    relay = Relay(drop_usage=True)
    timings.update(started=started, ttft_s=None)
    try:
        async with client.stream("POST", _PATH, json=body) as resp:
            if resp.status_code >= 400:
                raise _RoundFailed(_passed_on(app, replace(call, started=started), resp.status_code, await resp.aread(),
                                              resp.headers))
            async for chunk in resp.aiter_bytes():
                relay.feed(chunk)
                if timings["ttft_s"] is None and relay.started:
                    timings["ttft_s"] = time.monotonic() - started
    except httpx.HTTPError as e:
        raise _RoundFailed(_failed(app, replace(call, started=started), e)) from e
    return relay.completion()


async def _intercepted(app, call: Call, request: Request) -> Response:
    timings: dict = {}
    stream_options = {**(call.forward.get("stream_options") or {}), "include_usage": True}
    first = {**call.forward, "stream": True, "stream_options": stream_options}

    def log(body, c, rows, flags):
        save_call(app, replace(call, request=body, started=timings["started"]), c, state="ok", status=200,
                  ttft_s=timings["ttft_s"], rows=rows, extra_flags=flags)
        _note_unknown_cost(app, call, c)
    task = asyncio.ensure_future(answer(first, call.run, lambda body: _ask_round(app, call, body, timings), log))
    waited = 0.0
    try:
        while not task.done():
            step = min(_DISCONNECT_POLL_S, KEEPALIVE_S)
            await asyncio.wait({task}, timeout=step)
            waited += step
            if task.done():
                break
            if await request.is_disconnected():
                task.cancel()
                return Response(status_code=_STATUS_CLIENT_GONE)
            if call.request.get("stream") and waited >= KEEPALIVE_S:
                return StreamingResponse(_kept_alive(task, call), media_type="text/event-stream")
    except asyncio.CancelledError:
        task.cancel()
        raise
    try:
        return _finished(call, task.result())
    except _RoundFailed as e:
        return e.response


async def chat_completions(request: Request) -> Response:
    app, agent = request.app, request.path_params["agent"]
    if not valid_agent(agent):
        return JSONResponse(error_body("no such agent ID", "not_found"), status_code=404)
    try:
        body = json.loads(await request.body())
    except (ValueError, RecursionError):
        body = None
    if not isinstance(body, dict):
        return JSONResponse(error_body("the body must be a JSON object", "bad_request"), status_code=400)
    if app.state.runs.just_ended(agent):
        return JSONResponse(error_body("this agent's test run has just ended", "run_ended"), status_code=409)
    store = app.state.store
    safely(store, store.touch_agent, agent, live=agent not in app.state.active)
    call = resolve(app, agent, body)
    if _refused_paid(app, call, request):
        return JSONResponse(error_body(f"{call.model} is a paid model that isn't on this agent's list. Add it on "
                                       "ProvingBench's Models page.", "paid_model"), status_code=403)
    if call.run:
        return await _intercepted(app, call, request)
    return await (_live_stream(app, call) if body.get("stream") else _live_plain(app, call))


async def list_models(request: Request) -> Response:
    if not valid_agent(request.path_params["agent"]):
        return JSONResponse(error_body("no such agent ID", "not_found"), status_code=404)
    rows = await request.app.state.ups.listing()
    return JSONResponse({"object": "list", "data": [{"id": m["id"], "object": "model", "owned_by": m["upstream"]}
                                                    for m in rows]})
