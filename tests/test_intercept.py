import asyncio
import json

import httpx
import pytest
from fakes import FakeUpstream, event, sse, sse_call, sse_calls

from provingbench import proxy
from provingbench.app import create_app
from provingbench.intercept import BLOCKED, ActiveRun
from provingbench.stream import Relay

URL = "/a/bot/v1/chat/completions"
TOOL = {"type": "function", "function": {"name": "system_report", "parameters": {"type": "object", "properties": {}}}}


def chat(stream=True, **more):
    return {"model": "ollama/m", "stream": stream, "tools": [TOOL],
            "messages": [{"role": "user", "content": "hi"}], **more}


@pytest.fixture
def upstream():
    return FakeUpstream()


@pytest.fixture
async def app(cfg, upstream):
    app = create_app(cfg, upstream_transport=upstream.transport())
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7300") as c:
        yield c


def sse_calls_with_cost(name: str, cost: float) -> httpx.Response:
    parts = [{"index": 0, "id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}]
    body = event({"choices": [{"delta": {"tool_calls": parts}}]}) + event({"choices": [], "usage": {"cost": cost}})
    return httpx.Response(200, content=body + b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"})


def activate(app, stubs=None, allow=None):
    app.state.active["bot"] = ActiveRun(1, task_id="t", attempt_id="a1", stubs=stubs or {}, allow=allow or [])


def attempt_calls(app):
    return app.state.store.attempt_calls("a1")


async def test_a_stubbed_call_is_answered_by_the_proxy_and_the_agent_sees_only_the_final_reply(app, client, upstream):
    activate(app, stubs={"system_report": "running"})
    upstream.script(sse_call("system_report"), sse("It is running."))
    resp = await client.post(URL, json=chat())
    assert "It is running." in resp.text and "tool_calls" not in resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    sent = upstream.bodies[1]["messages"]
    assert sent[-1] == {"role": "tool", "tool_call_id": "c1", "content": "running"}
    assert sent[-2]["role"] == "assistant" and sent[-2]["tool_calls"][0]["function"]["name"] == "system_report"
    rows = attempt_calls(app)
    assert [t["answered"] for t in rows[0]["tool_calls"]] == ["stubbed"] and rows[1]["tool_calls"] == []
    assert rows[0]["run_id"] == 1 and rows[0]["task_id"] == "t"


async def test_anything_not_allowed_or_stubbed_is_blocked(app, client, upstream):
    activate(app)
    upstream.script(sse_call("docker_restart"), sse("I could not."))
    resp = await client.post(URL, json=chat())
    assert upstream.bodies[1]["messages"][-1]["content"] == BLOCKED and "docker_restart" not in resp.text
    assert attempt_calls(app)[0]["tool_calls"][0]["answered"] == "blocked"


async def test_a_blocked_unknown_tool_is_flagged(app, client, upstream):
    activate(app)
    upstream.script(sse_call("docker_restart"), sse("no"))
    await client.post(URL, json=chat())
    assert attempt_calls(app)[0]["flags"][0] == {"name": "unknown_tool", "detail": "docker_restart"}


async def test_an_allowed_call_reaches_the_agent(app, client, upstream):
    activate(app, allow=["system_report"])
    upstream.script(sse_call("system_report"))
    resp = await client.post(URL, json=chat())
    assert "system_report" in resp.text and len(upstream.bodies) == 1
    assert attempt_calls(app)[0]["tool_calls"][0]["answered"] == "real"


async def test_a_mixed_response_reaches_the_agent_not_at_all(app, client, upstream):
    activate(app, allow=["list_things"], stubs={"system_report": "ok"})
    upstream.script(sse_calls(("list_things", "{}"), ("danger", "{}")), sse("done"))
    resp = await client.post(URL, json=chat())
    assert "danger" not in resp.text and "list_things" not in resp.text
    assert [m["content"] for m in upstream.bodies[1]["messages"][-2:]] == [BLOCKED, BLOCKED]


async def test_eight_rounds_then_stop_with_a_flag(app, client, upstream):
    activate(app, stubs={"system_report": "x"})
    upstream.script_forever(sse_call("system_report"))
    resp = await client.post(URL, json=chat())
    assert "8 tool rounds" in resp.text and len(upstream.bodies) == 8
    assert "rounds_exhausted" in {f["name"] for c in attempt_calls(app) for f in c["flags"]}


async def test_a_non_streaming_agent_gets_json(app, client, upstream):
    activate(app)
    upstream.script(sse("Hi"))
    resp = await client.post(URL, json=chat(stream=False))
    assert resp.json()["choices"][0]["message"]["content"] == "Hi"
    assert upstream.bodies[0]["stream"] is True


async def test_upstream_failure_during_a_round_reaches_the_agent(app, client, upstream):
    activate(app)
    upstream.script(httpx.Response(500, json={"error": "boom"}))
    resp = await client.post(URL, json=chat())
    assert resp.status_code == 500 and resp.json() == {"error": "boom"}
    assert {f["name"] for f in attempt_calls(app)[0]["flags"]} == {"upstream_error"}


async def test_an_unreachable_upstream_during_a_round_is_a_502(app, client, upstream):
    activate(app)
    upstream.script(httpx.ConnectError("refused"))
    assert (await client.post(URL, json=chat())).status_code == 502


async def test_a_stream_breaking_during_a_round_is_a_502(app, client, upstream):
    activate(app, stubs={"system_report": "x"})
    upstream.script(sse_call("system_report"), httpx.ReadError("gone"))
    assert (await client.post(URL, json=chat())).status_code == 502


async def test_empty_answers_are_passed_on_and_flagged(app, client, upstream):
    activate(app)
    upstream.script(sse(""))
    await client.post(URL, json=chat())
    assert {f["name"] for f in attempt_calls(app)[0]["flags"]} == {"empty_reply"}


async def test_live_traffic_on_another_agent_is_untouched(app, client, upstream):
    activate(app)
    upstream.script(sse_call("danger"))
    assert "danger" in (await client.post("/a/other/v1/chat/completions", json=chat())).text


async def test_run_calls_do_not_count_as_live(app, client, upstream):
    activate(app)
    upstream.script(sse("y"))
    await client.post(URL, json=chat())
    assert app.state.store.last_live_at("bot") is None


# Buffered rounds: the agent must not time out, and must be able to give up


async def test_a_slow_round_sends_keep_alives_then_the_answer(app, client, upstream, monkeypatch):
    monkeypatch.setattr(proxy, "KEEPALIVE_S", 0.01)
    activate(app)
    upstream.hold()
    upstream.script(sse("Hi"))
    asyncio.get_running_loop().call_later(0.05, upstream.release)
    resp = await client.post(URL, json=chat())
    assert resp.text.startswith(": keep-alive") and "Hi" in resp.text and resp.status_code == 200


async def test_a_round_failing_after_keep_alives_reports_an_error_in_the_stream(app, client, upstream, monkeypatch):
    monkeypatch.setattr(proxy, "KEEPALIVE_S", 0.01)
    activate(app)
    upstream.hold()
    upstream.script(httpx.Response(500, json={"error": "boom"}))
    asyncio.get_running_loop().call_later(0.05, upstream.release)
    relay = Relay(drop_usage=False)
    resp = await client.post(URL, json=chat())
    relay.feed(resp.content)
    assert relay.completion().text == "" and b'"error"' in resp.content


async def test_an_agent_giving_up_stops_the_rounds(app, upstream):
    activate(app, stubs={"system_report": "x"})
    upstream.hold()
    upstream.script_forever(sse_call("system_report"))
    body = json.dumps(chat(stream=False)).encode()
    calls = []

    async def receive():
        calls.append(1)
        if len(calls) == 1:
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "POST", "path": URL, "raw_path": URL.encode(),
             "query_string": b"", "headers": [(b"host", b"127.0.0.1:7300"), (b"content-type", b"application/json")],
             "server": ("127.0.0.1", 7300), "client": ("127.0.0.1", 1), "scheme": "http", "http_version": "1.1"}
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    upstream.release()
    await asyncio.sleep(0.05)
    assert len(upstream.bodies) == 1


# A run's rules are fixed when the request arrives


async def test_a_round_keeps_the_attempt_it_started_in(app, client, upstream):
    activate(app, stubs={"system_report": "from attempt one"})
    upstream.hold()
    upstream.script(sse_call("system_report"), sse("done"))
    posting = asyncio.create_task(client.post(URL, json=chat()))
    await asyncio.sleep(0.02)
    app.state.active["bot"].attempt_id, app.state.active["bot"].stubs = "a2", {"system_report": "from attempt two"}
    upstream.release()
    await posting
    assert upstream.bodies[1]["messages"][-1]["content"] == "from attempt one"
    assert len(app.state.store.attempt_calls("a1")) == 2 and app.state.store.attempt_calls("a2") == []


async def test_a_runs_own_model_beats_any_selection(app, client, upstream):
    activate(app)
    app.state.active["bot"].model = "ollama/run-model"
    app.state.store.select("bot", "ollama/other", "owner")
    upstream.script(sse("x"))
    await client.post(URL, json=chat())
    assert upstream.bodies[0]["model"] == "run-model"


async def test_spending_is_checked_every_round(app, client, upstream):
    activate(app, stubs={"system_report": "x"})
    app.state.active["bot"].limit_usd = 0.5
    app.state.store.select("bot", "openrouter/x/y", "owner")
    upstream.script_forever(sse_calls_with_cost("system_report", 0.25))
    resp = await client.post(URL, json=chat())
    assert resp.status_code == 402 and len(upstream.bodies) == 2
    assert app.state.active["bot"].shared.over_limit


async def test_a_paid_call_with_no_known_cost_stops_the_spending(app, client, upstream):
    activate(app, stubs={"system_report": "x"})
    app.state.active["bot"].limit_usd = 0.5
    app.state.store.select("bot", "openrouter/never/priced", "owner")
    upstream.script_forever(sse_call("system_report"))
    resp = await client.post(URL, json=chat())
    assert resp.status_code == 402 and len(upstream.bodies) == 1
    assert "cost" in app.state.active["bot"].shared.over_limit


async def test_a_closed_run_stops_asking_the_model(app, client, upstream):
    activate(app, stubs={"system_report": "x"})
    upstream.hold()
    upstream.script_forever(sse_call("system_report"))
    posting = asyncio.create_task(client.post(URL, json=chat()))
    await asyncio.sleep(0.02)
    app.state.active["bot"].shared.closed = True
    upstream.release()
    resp = await posting
    assert resp.status_code == 409 and len(upstream.bodies) == 1
