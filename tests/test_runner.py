import asyncio
import sqlite3
from dataclasses import replace

import httpx
import pytest
from fakes import FakeAgent, FakeUpstream, event, sse, sse_call

from provingbench.app import create_app
from provingbench.auth import load_tokens
from provingbench.runner import RunManager, RunRefused, RunSpec

TARGET = """
id = "bot-test"
agent_id = "bot"
url = "http://agent.test/turn"
body = { session_id = "{session}", text = "{message}" }
reply_path = "reply"
"""
ONE_TURN = """
id = "hello"
title = "Says hello"
tags = ["chat"]
[[turns]]
say = "hi"
[stubs]
system_report = "running"
[[checks]]
type = "reply_not_empty"
"""
TWO_TURNS = """
id = "twice"
title = "Two turns"
[[turns]]
say = "first"
[[turns]]
say = "second"
[[checks]]
type = "reply_not_empty"
"""


class World:
    def __init__(self, app, upstream, agent):
        self.app, self.upstream, self.agent = app, upstream, agent
        self.store, self.runs = app.state.store, app.state.runs


def spec(models=("ollama/a",), suite="basic", repeats=1, max_cost_usd=1.0, label=""):
    return RunSpec("bot-test", suite, list(models), repeats, max_cost_usd, label)


@pytest.fixture
async def world(cfg, tmp_path):
    (cfg.data_dir / "targets").mkdir()
    (cfg.data_dir / "targets" / "bot.toml").write_text(TARGET, encoding="utf-8")
    for name, text in (("basic", ONE_TURN), ("two_turns", TWO_TURNS)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "task.toml").write_text(text, encoding="utf-8")
    full = replace(cfg, suites={"basic": tmp_path / "basic", "two_turns": tmp_path / "two_turns"})
    upstream, agent = FakeUpstream(), FakeAgent()
    app = create_app(full, upstream_transport=upstream.transport(), trigger_transport=agent.transport())
    agent.proxy = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7300")
    async with app.router.lifespan_context(app):
        yield World(app, upstream, agent)
    await agent.proxy.aclose()


async def run_to_end(world, run_spec):
    run_id = world.runs.start(run_spec)
    await world.runs.wait(run_id)
    return run_id


async def test_a_run_grades_every_model_task_and_repeat(world):
    world.upstream.script_forever(sse("hello"))
    run_id = await run_to_end(world, spec(models=["ollama/a", "ollama/b"], repeats=2))
    results = world.store.results(run_id)
    assert len(results) == 4 and {r["model"] for r in results} == {"ollama/a", "ollama/b"}
    assert {r["outcome"] for r in results} == {"pass"} and world.store.run(run_id)["state"] == "done"
    assert results[0]["checks"][0]["type"] == "reply_not_empty" and results[0]["tags"] == ["chat"]
    assert {b["model"] for b in world.upstream.bodies} == {"a", "b"}


async def test_calls_are_tied_to_their_attempt_and_stubs_apply(world):
    world.upstream.script(sse_call("system_report"), sse("running fine"))
    run_id = await run_to_end(world, spec())
    (result,) = world.store.results(run_id)
    calls = world.store.attempt_calls(result["attempt_id"])
    assert [c["task_id"] for c in calls] == ["hello", "hello"] and calls[0]["tool_calls"][0]["answered"] == "stubbed"
    assert result["reply"] == "running fine" and world.agent.ran_tools == []


async def test_each_repeat_gets_a_fresh_session_and_multi_turn_shares_one(world):
    world.upstream.script_forever(sse("hello"))
    await run_to_end(world, spec(suite="two_turns", repeats=2))
    sessions = world.agent.sessions
    assert len(sessions) == 4 and sessions[0] == sessions[1] and sessions[2] == sessions[3] and sessions[0] != sessions[2]


async def test_a_tasks_own_repeats_win(world, tmp_path):
    (tmp_path / "basic" / "task.toml").write_text(ONE_TURN.replace('tags = ["chat"]', 'tags = ["chat"]\nrepeats = 2'), encoding="utf-8")
    world.upstream.script_forever(sse("hello"))
    assert len(world.store.results(await run_to_end(world, spec(repeats=5)))) == 2


async def test_selection_is_restored_afterwards_even_on_failure(world):
    world.store.select("bot", "ollama/before", "owner")
    world.agent.crash_on_turn = True
    run_id = await run_to_end(world, spec())
    assert world.store.selection("bot") == "ollama/before" and world.store.results(run_id)[0]["outcome"] == "error"
    assert world.app.state.active == {}


async def test_no_earlier_selection_is_restored_as_none(world):
    world.upstream.script_forever(sse("hello"))
    await run_to_end(world, spec())
    assert world.store.selection("bot") is None


# Review Focus 1
async def test_a_killed_run_is_interrupted_and_restored_at_next_start(world):
    store = world.store
    run_id = store.create_run("bot-test", "bot", "basic", ["ollama/test"], 1, 1.0, "", "ollama/before", 1)
    store.select("bot", "ollama/test", "run")      # as a killed process leaves things
    later = RunManager(store, world.app.state.cfg, {})
    later.recover()
    assert store.run(run_id)["state"] == "interrupted" and store.selection("bot") == "ollama/before"


async def test_a_run_saves_the_selection_it_will_restore(world):
    world.store.select("bot", "ollama/before", "owner")
    world.upstream.hold()
    world.upstream.script_forever(sse("hi"))
    run_id = world.runs.start(spec())
    assert world.store.run(run_id)["saved_selection"] == "ollama/before"
    world.upstream.release()
    await world.runs.wait(run_id)


async def test_recent_live_traffic_refuses_the_run(world):
    world.store.touch_agent("bot", live=True)
    with pytest.raises(RunRefused, match="live"):
        world.runs.start(spec())


async def test_old_live_traffic_does_not_refuse(world):
    world.store.conn.execute("INSERT INTO agents (id, first_seen, last_seen, last_live_at) VALUES "
                             "('bot', '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00')")
    world.upstream.script_forever(sse("hi"))
    await run_to_end(world, spec())


async def test_an_unreachable_target_fails_the_run_at_once_with_a_message(world):
    world.agent.down = True
    run_id = await run_to_end(world, spec(repeats=3))
    run = world.store.run(run_id)
    assert run["state"] == "failed" and "reach" in run["error"] and world.store.results(run_id) == []


async def test_one_run_per_target(world):
    world.upstream.hold()
    world.upstream.script_forever(sse("hi"))
    run_id = world.runs.start(spec())
    with pytest.raises(RunRefused, match="already"):
        world.runs.start(spec())
    world.upstream.release()
    await world.runs.wait(run_id)


@pytest.mark.parametrize("bad,problem", [(dict(suite="nope"), "suite"), (dict(models=[]), "model")])
async def test_a_bad_request_is_refused(world, bad, problem):
    with pytest.raises(RunRefused, match=problem):
        world.runs.start(spec(**bad))


async def test_an_unknown_target_is_refused(world):
    with pytest.raises(RunRefused, match="target"):
        world.runs.start(replace(spec(), target="nope"))


async def test_a_bad_task_file_stops_the_run_before_it_starts(world, tmp_path):
    (tmp_path / "basic" / "bad.toml").write_text("id = ", encoding="utf-8")
    with pytest.raises(RunRefused, match="bad.toml"):
        world.runs.start(spec())
    assert world.store.runs() == []


async def test_spending_the_limit_stops_the_run_and_the_owner_can_resume(world):
    world.upstream.script_forever(sse("hi", usage={"cost": 0.6}))
    run_id = await run_to_end(world, spec(models=["openrouter/x/y"], repeats=4, max_cost_usd=1.0))
    assert world.store.run(run_id)["state"] == "stopped_cost" and len(world.store.results(run_id)) == 2
    world.runs.resume(run_id, max_cost_usd=5.0)
    await world.runs.wait(run_id)
    assert world.store.run(run_id)["state"] == "done" and len(world.store.results(run_id)) == 4
    assert world.store.spend(run_id) == pytest.approx(2.4)


async def test_a_stopped_run_keeps_its_results(world):
    world.upstream.script_forever(sse("hi"))
    world.agent.pause_on_turn = 3
    run_id = world.runs.start(spec(repeats=50))
    await world.agent.paused.wait()
    world.runs.stop(run_id)
    await world.runs.wait(run_id)
    assert world.store.run(run_id)["state"] == "stopped" and len(world.store.results(run_id)) == 2
    assert world.app.state.active == {}


async def test_stopping_a_run_that_has_not_begun_still_cleans_up(world):
    world.store.select("bot", "ollama/before", "owner")
    run_id = world.runs.start(spec())
    world.runs.stop(run_id)
    await world.runs.wait(run_id)
    assert world.store.run(run_id)["state"] == "stopped" and world.store.selection("bot") == "ollama/before"
    assert world.app.state.active == {}


async def test_results_store_the_task_hash_and_the_label(world):
    world.upstream.script_forever(sse("hi"))
    run_id = await run_to_end(world, spec(label="v2"))
    assert len(world.store.results(run_id)[0]["task_hash"]) == 64 and world.store.run(run_id)["label"] == "v2"


async def test_estimate_is_positive_for_paid_and_zero_for_local(world):
    world.store.save_models([{"id": "openrouter/x/y", "upstream": "openrouter", "price_prompt": 1e-6,
                              "price_completion": 2e-6, "context_length": None, "supports_tools": True}])
    assert world.runs.estimate(spec(models=["openrouter/x/y"])) == pytest.approx(2 * (2000 * 1e-6 + 300 * 2e-6))
    assert world.runs.estimate(spec(models=["ollama/m"])) == 0
    assert world.runs.estimate(spec(models=["openrouter/never/seen"])) is None


# Through the API


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def run_body(**over):
    return {"target": "bot-test", "suite": "basic", "models": ["ollama/a"], "repeats": 1, **over}


@pytest.fixture
async def api(world):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=world.app), base_url="http://127.0.0.1:7300") as c:
        yield c


@pytest.fixture
def tokens(cfg):
    return load_tokens(cfg.data_dir)


async def test_agent_token_cannot_test_a_paid_model_or_raise_the_limit(api, tokens):
    paid = run_body(models=["openrouter/x/y"])
    assert (await api.post("/api/v1/runs", json=paid, headers=bearer(tokens.agent))).status_code == 403
    over = run_body(max_cost_usd=50)
    assert (await api.post("/api/v1/runs", json=over, headers=bearer(tokens.agent))).status_code == 403


async def test_a_run_through_the_api(world, api, tokens):
    world.upstream.script_forever(sse("hi"))
    started = await api.post("/api/v1/runs", json=run_body(), headers=bearer(tokens.agent))
    assert started.status_code == 202
    run_id = started.json()["id"]
    await world.runs.wait(run_id)
    body = (await api.get(f"/api/v1/runs/{run_id}", headers=bearer(tokens.agent))).json()
    assert body["state"] == "done" and body["total"] == 1 and body["results"][0]["outcome"] == "pass"
    result = (await api.get(f"/api/v1/results/{body['results'][0]['id']}", headers=bearer(tokens.agent))).json()
    assert result["calls"][0]["response"]["text"] == "hi"


async def test_targets_suites_and_estimate_are_listed(api, tokens):
    head = bearer(tokens.owner)
    assert (await api.get("/api/v1/targets", headers=head)).json()["targets"][0]["id"] == "bot-test"
    assert {s["name"] for s in (await api.get("/api/v1/suites", headers=head)).json()["suites"]} == {"basic", "two_turns"}
    assert (await api.post("/api/v1/estimate", json=run_body(), headers=head)).json()["estimate_usd"] == 0


async def test_a_refused_run_is_a_400_with_the_reason(api, tokens):
    resp = await api.post("/api/v1/runs", json=run_body(suite="nope"), headers=bearer(tokens.owner))
    assert resp.status_code == 400 and "suite" in resp.json()["error"]["message"]


async def test_a_conflict_is_a_409(world, api, tokens):
    world.store.touch_agent("bot", live=True)
    assert (await api.post("/api/v1/runs", json=run_body(), headers=bearer(tokens.owner))).status_code == 409


@pytest.mark.parametrize("body", [{"target": "bot-test"}, run_body(models="ollama/a"), run_body(repeats=0),
                                  run_body(max_cost_usd=-1), run_body(models=[5])])
async def test_a_malformed_run_is_a_400(api, tokens, body):
    assert (await api.post("/api/v1/runs", json=body, headers=bearer(tokens.owner))).status_code == 400


async def test_only_the_owner_resumes(world, api, tokens):
    world.upstream.script_forever(sse("hi"))
    run_id = await run_to_end(world, spec())
    path = f"/api/v1/runs/{run_id}/resume"
    assert (await api.post(path, json={"max_cost_usd": 5}, headers=bearer(tokens.agent))).status_code == 403


async def test_an_agent_may_stop_a_run(world, api, tokens):
    world.upstream.script_forever(sse("hi"))
    run_id = world.runs.start(spec(repeats=50))
    resp = await api.post(f"/api/v1/runs/{run_id}/stop", json={}, headers=bearer(tokens.agent))
    assert resp.status_code == 200
    await world.runs.wait(run_id)
    assert world.store.run(run_id)["state"] == "stopped"


async def test_shutting_down_mid_run_marks_it_interrupted_and_restores_the_model(world):
    world.store.select("bot", "ollama/before", "owner")
    world.upstream.hold()
    world.upstream.script_forever(sse("hi"))
    run_id = world.runs.start(spec())
    await asyncio.sleep(0.05)
    await world.runs.aclose()
    assert world.store.run(run_id)["state"] == "interrupted" and world.store.selection("bot") == "ollama/before"


# A wrongly pointed target must not pass


async def test_an_agent_that_never_calls_the_proxy_fails_the_run_and_earns_no_grade(world):
    world.agent.skips_proxy = True
    run_id = await run_to_end(world, spec(repeats=3))
    run = world.store.run(run_id)
    assert run["state"] == "failed" and "base URL" in run["error"] and world.store.results(run_id) == []
    assert len(world.agent.sessions) == 1


# After a run the agent may still be finishing a turn


async def test_after_a_run_the_agents_late_calls_are_refused(world):
    world.upstream.script_forever(sse_call("rm_rf"))
    world.agent.pause_on_turn = 1
    run_id = world.runs.start(spec())
    await world.agent.paused.wait()
    world.runs.stop(run_id)
    await world.runs.wait(run_id)
    calls_before = len(world.upstream.bodies)
    resp = await world.agent.proxy.post("/a/bot/v1/chat/completions", json={"model": "ollama/a", "stream": True, "messages": []})
    assert resp.status_code == 409 and "rm_rf" not in resp.text and len(world.upstream.bodies) == calls_before
    assert world.store.last_live_at("bot") is None


async def test_the_grace_period_ends(world, monkeypatch):
    world.upstream.script_forever(sse("hi"))
    await run_to_end(world, spec())
    monkeypatch.setattr(world.runs, "_ended", {"bot": 0.0})
    resp = await world.agent.proxy.post("/a/bot/v1/chat/completions", json={"model": "ollama/a", "stream": True, "messages": []})
    assert resp.status_code == 200


async def test_a_new_run_ends_the_grace_period(world):
    world.upstream.script_forever(sse("hi"))
    await run_to_end(world, spec())
    assert len(world.store.results(await run_to_end(world, spec()))) == 1


async def test_late_calls_are_refused_after_a_crash_too(world):
    store = world.store
    store.create_run("bot-test", "bot", "basic", ["ollama/test"], 1, 1.0, "", None, 1)
    world.runs.recover()
    assert world.runs.just_ended("bot")


# Trouble with the log


async def test_a_logging_failure_makes_an_attempt_ungradable_not_failed(world, monkeypatch):
    def locked(_):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(world.store, "add_call", locked)
    world.upstream.script_forever(sse("hi"))
    run_id = await run_to_end(world, spec())
    assert world.store.results(run_id)[0]["outcome"] == "error" and "log" in world.store.results(run_id)[0]["error"]


async def test_a_database_failure_restoring_the_model_still_ends_the_run(world, monkeypatch):
    world.upstream.script_forever(sse("hi"))
    real = world.store.select

    def fails_for_run_restore(agent, model, by):
        if model is None:
            raise sqlite3.OperationalError("database is locked")
        real(agent, model, by)
    monkeypatch.setattr(world.store, "select", fails_for_run_restore)
    run_id = await run_to_end(world, spec())
    assert world.store.run(run_id)["state"] == "done"


async def test_a_run_row_with_nothing_behind_it_reads_as_interrupted_and_can_resume(world, api, tokens):
    run_id = world.store.create_run("bot-test", "bot", "basic", ["ollama/a"], 1, 1.0, "", None, 1)
    body = (await api.get(f"/api/v1/runs/{run_id}", headers=bearer(tokens.owner))).json()
    assert body["state"] == "interrupted"
    world.upstream.script_forever(sse("hi"))
    world.runs.resume(run_id)
    await world.runs.wait(run_id)
    assert world.store.run(run_id)["state"] == "done"


# Costs


async def test_the_limit_holds_inside_one_attempt_and_the_cut_attempt_earns_no_grade(world):
    def costly():
        parts = [{"index": 0, "id": "c1", "type": "function", "function": {"name": "system_report", "arguments": "{}"}}]
        body = event({"choices": [{"delta": {"tool_calls": parts}}]}) + event({"choices": [], "usage": {"cost": 0.25}})
        return httpx.Response(200, content=body + b"data: [DONE]\n\n")
    world.upstream.script_forever(costly())
    run_id = await run_to_end(world, spec(models=["openrouter/x/y"], max_cost_usd=0.5))
    assert world.store.run(run_id)["state"] == "stopped_cost" and world.store.spend(run_id) == pytest.approx(0.5)
    assert world.store.results(run_id) == []


async def test_a_paid_model_with_no_known_cost_stops_the_run_and_says_why(world):
    world.upstream.script_forever(sse("hi"))
    run_id = await run_to_end(world, spec(models=["openrouter/never/priced"], repeats=5))
    run = world.store.run(run_id)
    assert run["state"] == "stopped_cost" and "cost" in run["error"] and len(world.upstream.bodies) == 1


# The switch can't move under a run


async def test_switching_a_running_agent_is_refused(world, api, tokens):
    world.upstream.hold()
    world.upstream.script_forever(sse("hi"))
    run_id = world.runs.start(spec())
    resp = await api.put("/api/v1/agents/bot/model", json={"model": "ollama/z"}, headers=bearer(tokens.owner))
    assert resp.status_code == 409
    world.upstream.release()
    await world.runs.wait(run_id)


async def test_an_agent_cannot_reach_a_paid_judge_through_a_run(world, api, tokens, tmp_path, cfg):
    (tmp_path / "basic" / "task.toml").write_text(
        ONE_TURN + '[[checks]]\ntype = "judge"\nrubric = "r"\n', encoding="utf-8")
    world.app.state.cfg = replace(world.app.state.cfg, judge_model="openrouter/x/judge")
    resp = await api.post("/api/v1/runs", json=run_body(), headers=bearer(tokens.agent))
    assert resp.status_code == 403 and "judge" in resp.json()["error"]["message"]
