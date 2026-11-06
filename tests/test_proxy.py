import asyncio
import json
import sqlite3

import httpx
import pytest
from fakes import Breaks, FakeUpstream, json_reply, sse

from provingbench.app import create_app

URL = "/a/bot/v1/chat/completions"
JSON = {"Content-Type": "application/json"}


def chat(stream=True, model="ollama/m", **more):
    return {"model": model, "stream": stream, "messages": [{"role": "user", "content": "hi"}], **more}


@pytest.fixture
def upstream():
    return FakeUpstream()


@pytest.fixture
async def app(cfg, upstream):
    app = create_app(cfg, upstream_transport=upstream.transport())
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def proxy(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7300") as c:
        yield c


def only_call(app):
    (call,) = app.state.store.calls()
    return call


def flag_names(call):
    return {f["name"] for f in call["flags"]}


async def test_stream_passes_through_and_is_logged(app, proxy, upstream):
    upstream.script(sse("Hi", usage={"prompt_tokens": 3, "completion_tokens": 1}))
    resp = await proxy.post(URL, json=chat())
    assert resp.status_code == 200 and "Hi" in resp.text and '"usage"' not in resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    call = only_call(app)
    assert call["agent_id"] == "bot" and call["model"] == "ollama/m" and call["state"] == "ok"
    assert call["upstream"] == "ollama" and call["status"] == 200 and call["response"]["text"] == "Hi"
    assert call["prompt_tokens"] == 3 and call["ttft_s"] is not None and call["cost_usd"] == 0
    assert call["request"]["messages"][0]["content"] == "hi"


async def test_upstream_is_asked_for_usage(proxy, upstream):
    upstream.script(sse("Hi"))
    await proxy.post(URL, json=chat())
    assert upstream.bodies[-1]["stream_options"]["include_usage"] is True


async def test_agent_that_asked_for_usage_still_gets_it(proxy, upstream):
    upstream.script(sse("Hi", usage={"prompt_tokens": 3}))
    resp = await proxy.post(URL, json=chat(stream_options={"include_usage": True}))
    assert '"usage"' in resp.text


async def test_selection_overrides_the_request_model(app, proxy, upstream):
    app.state.store.select("bot", "openrouter/x/y", "owner")
    upstream.script(sse("Hi"))
    await proxy.post(URL, json=chat(model="ollama/m"))
    assert upstream.requests[-1].url.host == "openrouter.ai" and upstream.bodies[-1]["model"] == "x/y"
    assert upstream.bodies[-1]["usage"] == {"include": True} and "logprobs" not in upstream.bodies[-1]
    assert only_call(app)["model"] == "openrouter/x/y"


async def test_without_a_selection_a_bare_model_goes_to_the_default_upstream(app, proxy, upstream):
    upstream.script(sse("Hi"))
    await proxy.post(URL, json=chat(model="qwen3:8b"))
    assert upstream.bodies[-1]["model"] == "qwen3:8b" and only_call(app)["model"] == "ollama/qwen3:8b"


async def test_other_settings_are_forwarded_as_sent(proxy, upstream):
    upstream.script(sse("Hi"))
    await proxy.post(URL, json=chat(max_tokens=7, reasoning_effort="low", tools=[{"type": "function"}]))
    body = upstream.bodies[-1]
    assert body["max_tokens"] == 7 and body["reasoning_effort"] == "low" and body["tools"] == [{"type": "function"}]


async def test_agent_key_is_not_forwarded_or_logged(app, proxy, upstream, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "real-key")
    app.state.store.select("bot", "openrouter/x/y", "owner")
    upstream.script(sse("Hi"))
    await proxy.post(URL, json=chat(), headers={"Authorization": "Bearer agent-secret"})
    assert upstream.requests[-1].headers["authorization"] == "Bearer real-key"
    text = json.dumps(only_call(app))
    assert "agent-secret" not in text and "real-key" not in text


async def test_upstream_error_keeps_status_and_body_and_is_flagged(app, proxy, upstream):
    upstream.script(httpx.Response(429, json={"error": "slow down"}, headers={"Retry-After": "3"}))
    resp = await proxy.post(URL, json=chat())
    assert resp.status_code == 429 and resp.json() == {"error": "slow down"} and resp.headers["retry-after"] == "3"
    call = only_call(app)
    assert flag_names(call) == {"upstream_error"} and call["state"] == "error" and call["status"] == 429


async def test_unreachable_upstream_is_a_502_in_openai_form(app, proxy, upstream):
    upstream.script(httpx.ConnectError("refused"))
    resp = await proxy.post(URL, json=chat())
    assert resp.status_code == 502 and "refused" in resp.json()["error"]["message"]
    assert flag_names(only_call(app)) == {"upstream_error"}


async def test_non_stream_is_logged_too(app, proxy, upstream):
    upstream.script(json_reply("Hi"))
    assert (await proxy.post(URL, json=chat(stream=False))).json()["choices"][0]["message"]["content"] == "Hi"
    assert only_call(app)["response"]["text"] == "Hi" and "stream_options" not in upstream.bodies[-1]


async def test_flags_are_recorded(app, proxy, upstream):
    upstream.script(sse(""))
    await proxy.post(URL, json=chat())
    assert flag_names(only_call(app)) == {"empty_reply"}


async def test_bad_json_is_a_400_and_a_bad_agent_id_a_404(proxy):
    assert (await proxy.post(URL, content=b"{", headers=JSON)).status_code == 400
    assert (await proxy.post(URL, content=b"[1]", headers=JSON)).status_code == 400
    assert (await proxy.post("/a/bad%20id/v1/chat/completions", json=chat())).status_code == 404


async def test_ollama_gets_logprobs(proxy, upstream):
    upstream.script(sse("Hi"))
    await proxy.post(URL, json=chat())
    assert upstream.bodies[-1]["logprobs"] is True


async def test_live_traffic_is_marked_live(app, proxy, upstream):
    upstream.script(sse("Hi"))
    await proxy.post(URL, json=chat())
    assert app.state.store.last_live_at("bot")


async def test_models_are_listed_for_the_agent(proxy):
    body = (await proxy.get("/a/bot/v1/models")).json()
    assert body["object"] == "list" and body["data"][0] == {"id": "ollama/m", "object": "model", "owned_by": "ollama"}


# Review Focus 2
async def test_agent_hanging_up_mid_stream_logs_cancelled_and_closes_upstream(app, upstream):
    upstream.endless("par")
    sent, got_first = [], []
    body = json.dumps(chat()).encode()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        while not got_first:
            await asyncio.sleep(0.01)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            got_first.append(True)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "POST", "path": URL, "raw_path": URL.encode(),
             "query_string": b"", "headers": [(b"host", b"127.0.0.1:7300"), (b"content-type", b"application/json")],
             "server": ("127.0.0.1", 7300), "client": ("127.0.0.1", 1), "scheme": "http", "http_version": "1.1"}
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    call = only_call(app)
    assert call["state"] == "cancelled" and call["response"]["text"].startswith("par") and upstream.closed


# Review Focus 4
async def test_a_locked_database_never_fails_the_call(app, proxy, upstream, monkeypatch):
    def locked(_):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(app.state.store, "add_call", locked)
    upstream.script(sse("Hi"))
    assert (await proxy.post(URL, json=chat())).status_code == 200
    assert app.state.store.log_errors == 1


async def test_a_locked_database_on_touch_never_fails_the_call(app, proxy, upstream, monkeypatch):
    def locked(*_, **__):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(app.state.store, "touch_agent", locked)
    upstream.script(sse("Hi"))
    assert (await proxy.post(URL, json=chat())).status_code == 200
    assert app.state.store.log_errors == 1


async def test_a_locked_database_on_selection_never_fails_the_call(app, proxy, upstream, monkeypatch):
    def locked(*_, **__):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(app.state.store, "selection", locked)
    upstream.script(sse("Hi"))
    assert (await proxy.post(URL, json=chat())).status_code == 200
    assert upstream.bodies[-1]["model"] == "m"


# Paid models need the owner's say-so, however the agent asks for them


async def test_naming_a_paid_model_without_being_on_the_list_is_refused(app, proxy, upstream):
    resp = await proxy.post(URL, json=chat(model="openrouter/x/expensive"))
    assert resp.status_code == 403 and "paid model" in resp.json()["error"]["message"] and upstream.requests == []


async def test_a_paid_model_on_the_list_goes_through(app, proxy, upstream):
    app.state.store.set_allowed("bot", ["openrouter/x/expensive"])
    upstream.script(sse("Hi"))
    assert (await proxy.post(URL, json=chat(model="openrouter/x/expensive"))).status_code == 200


async def test_a_listed_free_cloud_model_goes_through(app, proxy, upstream):
    app.state.store.save_models([{"id": "openrouter/free/one", "upstream": "openrouter", "price_prompt": 0,
                                  "price_completion": 0, "context_length": None, "supports_tools": None}])
    upstream.script(sse("Hi"))
    assert (await proxy.post(URL, json=chat(model="openrouter/free/one"))).status_code == 200


# An upstream dying mid-stream must look like a broken stream to the agent, not a finished one


async def test_a_stream_breaking_midway_breaks_for_the_agent_too(app, proxy, upstream):
    upstream.script(httpx.Response(200, stream=Breaks(), headers={"content-type": "text/event-stream"}))
    with pytest.raises(httpx.ReadError):
        await proxy.post(URL, json=chat())
    call = only_call(app)
    assert call["state"] == "error" and flag_names(call) == {"upstream_error"} and call["response"]["text"] == "par"


async def test_listing_models_survives_a_locked_database(app, proxy, monkeypatch):
    def locked(_):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(app.state.store, "save_models", locked)
    assert (await proxy.get("/a/bot/v1/models")).status_code == 200
