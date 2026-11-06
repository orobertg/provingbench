import httpx
import pytest
from conftest import record
from fakes import FakeUpstream

from provingbench.app import create_app
from provingbench.auth import load_tokens
from provingbench.store import Flag


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def app(cfg):
    app = create_app(cfg, upstream_transport=FakeUpstream().transport())
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7300") as c:
        yield c


@pytest.fixture
def tokens(cfg):
    return load_tokens(cfg.data_dir)


@pytest.fixture
def owner(tokens):
    return bearer(tokens.owner)


@pytest.fixture
def agent(tokens):
    return bearer(tokens.agent)


async def test_models_are_listed(client, agent):
    assert (await client.get("/api/v1/models", headers=agent)).json()["models"][0]["id"] == "ollama/m"


async def test_agent_may_switch_to_local_or_free(app, client, agent):
    resp = await client.put("/api/v1/agents/bot/model", json={"model": "ollama/m"}, headers=agent)
    assert resp.status_code == 200 and app.state.store.selections("bot")[-1]["by"] == "agent"
    assert resp.json() == {"model": "ollama/m", "cloud": False}


async def test_agent_may_not_switch_to_paid_unless_allowed(client, agent, owner):
    async def put():
        return await client.put("/api/v1/agents/bot/model", json={"model": "openrouter/x/y"}, headers=agent)
    assert (await put()).status_code == 403
    await client.put("/api/v1/agents/bot/allowed-models", json={"models": ["openrouter/x/y"]}, headers=owner)
    assert (await put()).status_code == 200


async def test_the_owner_may_switch_to_anything(client, owner):
    resp = await client.put("/api/v1/agents/bot/model", json={"model": "openrouter/x/y"}, headers=owner)
    assert resp.status_code == 200


async def test_only_the_owner_sets_allowed_models(client, agent):
    resp = await client.put("/api/v1/agents/bot/allowed-models", json={"models": []}, headers=agent)
    assert resp.status_code == 403


async def test_current_model_reports_a_cloud_warning(client, owner):
    await client.put("/api/v1/agents/bot/model", json={"model": "openrouter/x/y"}, headers=owner)
    body = (await client.get("/api/v1/agents/bot/model", headers=owner)).json()
    assert body == {"model": "openrouter/x/y", "cloud": True}


async def test_a_local_or_missing_model_is_not_cloud(client, owner):
    assert (await client.get("/api/v1/agents/bot/model", headers=owner)).json() == {"model": None, "cloud": False}
    await client.put("/api/v1/agents/bot/model", json={"model": "ollama/m"}, headers=owner)
    assert (await client.get("/api/v1/agents/bot/model", headers=owner)).json()["cloud"] is False


async def test_a_null_model_clears_the_selection(client, app, owner):
    await client.put("/api/v1/agents/bot/model", json={"model": "ollama/m"}, headers=owner)
    await client.put("/api/v1/agents/bot/model", json={"model": None}, headers=owner)
    assert app.state.store.selection("bot") is None


@pytest.mark.parametrize("body", [{"model": 5}, {}, [1]])
async def test_a_bad_switch_is_a_400(client, owner, body):
    assert (await client.put("/api/v1/agents/bot/model", json=body, headers=owner)).status_code == 400


async def test_a_bad_agent_id_is_a_404(client, owner):
    assert (await client.get("/api/v1/agents/bad%20id/model", headers=owner)).status_code == 404


async def test_agents_show_their_current_model(client, owner):
    await client.put("/api/v1/agents/bot/model", json={"model": "ollama/m"}, headers=owner)
    (row,) = (await client.get("/api/v1/agents", headers=owner)).json()["agents"]
    assert row["id"] == "bot" and row["model"] == "ollama/m"


async def test_calls_filter_and_detail(app, client, owner):
    store = app.state.store
    first = store.add_call(record(flags=[Flag("slow")]))
    second = store.add_call(record())
    assert [c["id"] for c in (await client.get(f"/api/v1/calls?after={first}", headers=owner)).json()["calls"]] == [second]
    assert [c["id"] for c in (await client.get("/api/v1/calls?flag=slow", headers=owner)).json()["calls"]] == [first]
    assert (await client.get(f"/api/v1/calls/{first}", headers=owner)).json()["id"] == first
    assert (await client.get("/api/v1/calls/999", headers=owner)).status_code == 404


async def test_calls_with_a_bad_number_is_a_400(client, owner):
    assert (await client.get("/api/v1/calls?after=x", headers=owner)).status_code == 400


async def test_status_reports_logging_errors(app, client, owner):
    app.state.store.log_errors = 2
    assert (await client.get("/api/v1/status", headers=owner)).json()["log_errors"] == 2


def _seed_result(app):
    store = app.state.store
    run_id = store.create_run("t", "bot", "s", ["m"], 1, 1.0, "", None, 1)
    return store.add_result(run_id, "m", "t", "h", [], 0, "a1", "fail", [], "r", None, 1.0, 0.0)


async def test_only_the_owner_overrides(app, client, agent, owner):
    rid = _seed_result(app)
    body = {"outcome": "pass", "note": "right"}
    assert (await client.post(f"/api/v1/results/{rid}/override", json=body, headers=agent)).status_code == 403
    assert (await client.post(f"/api/v1/results/{rid}/override", json=body, headers=owner)).status_code == 200


async def test_a_result_shows_both_grades(app, client, owner):
    rid = _seed_result(app)
    await client.post(f"/api/v1/results/{rid}/override", json={"outcome": "pass", "note": "right"}, headers=owner)
    body = (await client.get(f"/api/v1/results/{rid}", headers=owner)).json()
    assert body["outcome"] == "fail" and body["override"]["outcome"] == "pass" and body["override"]["note"] == "right"


async def test_a_result_with_no_override_says_none(app, client, owner):
    rid = _seed_result(app)
    assert (await client.get(f"/api/v1/results/{rid}", headers=owner)).json()["override"] is None


@pytest.mark.parametrize("body", [{"outcome": "maybe", "note": "x"}, {"outcome": "pass"}, {"outcome": "pass", "note": ""}])
async def test_a_bad_override_is_a_400(app, client, owner, body):
    rid = _seed_result(app)
    assert (await client.post(f"/api/v1/results/{rid}/override", json=body, headers=owner)).status_code == 400


async def test_overriding_a_missing_result_is_a_404(client, owner):
    resp = await client.post("/api/v1/results/999/override", json={"outcome": "pass", "note": "x"}, headers=owner)
    assert resp.status_code == 404


async def test_notes_appear_in_the_profile(client, owner, agent):
    assert (await client.put("/api/v1/models/ollama/m/notes", json={"text": "fast"}, headers=agent)).status_code == 403
    assert (await client.put("/api/v1/models/ollama/m/notes", json={"text": "fast"}, headers=owner)).status_code == 200
    assert (await client.get("/api/v1/models/ollama/m/profile", headers=agent)).json()["notes"] == "fast"
