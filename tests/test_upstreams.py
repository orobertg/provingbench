import httpx
import pytest

from provingbench.upstreams import Upstreams, cost_of, is_paid, route

OPENROUTER_MODELS = {"data": [
    {"id": "x/y", "context_length": 8000, "pricing": {"prompt": "0.000001", "completion": "0.000002"},
     "supported_parameters": ["tools", "temperature"]},
    {"id": "free/one", "context_length": 4000, "pricing": {"prompt": "0", "completion": "0"},
     "supported_parameters": []}], "links": {"next": None}}


def handler(openrouter=None):
    def handle(request):
        if request.url.host == "localhost":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return openrouter or httpx.Response(200, json=OPENROUTER_MODELS)
    return handle


def ups_for(cfg, store, **kw):
    return Upstreams(cfg, store, transport=httpx.MockTransport(handler(**kw)))


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")


def test_route(cfg):
    assert route(cfg, "ollama/qwen3:8b")[1] == "qwen3:8b"
    up, name = route(cfg, "openrouter/qwen/qwen3-235b")
    assert up.name == "openrouter" and name == "qwen/qwen3-235b"
    up, name = route(cfg, "plain-model")
    assert up.name == "ollama" and name == "plain-model"
    up, name = route(cfg, "qwen/qwen3")
    assert up.name == "ollama" and name == "qwen/qwen3"


async def test_listing_merges_with_details(cfg, store, key):
    ups = ups_for(cfg, store)
    rows = {m["id"]: m for m in await ups.listing()}
    assert rows["ollama/m"]["price_prompt"] == 0
    r = rows["openrouter/x/y"]
    assert r["context_length"] == 8000 and r["supports_tools"] and r["price_prompt"] == pytest.approx(1e-6)
    assert rows["openrouter/free/one"]["supports_tools"] is False
    await ups.aclose()


async def test_no_openrouter_key_shows_only_ollama(cfg, store):
    ups = ups_for(cfg, store)
    assert all(m["id"].startswith("ollama/") for m in await ups.listing())
    await ups.aclose()


async def test_one_upstream_down_keeps_the_other_and_reports_it(cfg, store, key):
    ups = ups_for(cfg, store, openrouter=httpx.Response(500, text="boom"))
    errors = await ups.refresh()
    assert "openrouter" in errors and any(m["id"].startswith("ollama/") for m in store.models())
    await ups.aclose()


async def test_listing_is_cached_for_an_hour(cfg, store):
    count = []

    def handle(request):
        count.append(1)
        return httpx.Response(200, json={"data": [{"id": "m"}]})
    ups = Upstreams(cfg, store, transport=httpx.MockTransport(handle))
    await ups.listing()
    await ups.listing()
    assert len(count) == 1
    await ups.aclose()


async def test_key_goes_out_only_to_its_own_upstream(cfg, store, key):
    seen = {}

    def handle(request):
        seen[request.url.host] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})
    ups = Upstreams(cfg, store, transport=httpx.MockTransport(handle))
    await ups.refresh()
    assert seen == {"localhost": None, "openrouter.ai": "Bearer k"}
    await ups.aclose()


@pytest.fixture
def priced(store):
    store.save_models([row("ollama/m", "ollama", 0, 0), row("openrouter/free/one", "openrouter", 0, 0),
                       row("openrouter/x/y", "openrouter", 1e-6, 2e-6)])
    return store


def row(id_, upstream, prompt, completion):
    return {"id": id_, "upstream": upstream, "price_prompt": prompt, "price_completion": completion,
            "context_length": None, "supports_tools": None}


def test_paid(cfg, priced):
    assert not is_paid(cfg, priced, "ollama/m")
    assert not is_paid(cfg, priced, "ollama/never-listed")
    assert not is_paid(cfg, priced, "openrouter/free/one")
    assert is_paid(cfg, priced, "openrouter/x/y")
    assert is_paid(cfg, priced, "openrouter/never/seen")     # unknown price counts as paid


def test_cost_prefers_the_reported_figure(cfg, priced):
    assert cost_of(cfg, priced, "openrouter/x/y", {"cost": 0.5, "prompt_tokens": 1}) == 0.5
    assert cost_of(cfg, priced, "openrouter/x/y", {"prompt_tokens": 1000, "completion_tokens": 0}) == pytest.approx(1e-3)
    assert cost_of(cfg, priced, "ollama/m", {"prompt_tokens": 9}) == 0
    assert cost_of(cfg, priced, "openrouter/x/y", None) is None
