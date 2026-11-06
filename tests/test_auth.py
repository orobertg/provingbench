from dataclasses import replace

import httpx
import pytest
from fakes import FakeUpstream

from provingbench.app import create_app
from provingbench.auth import load_tokens

URL = "/a/bot/v1/chat/completions"


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


async def client_for(cfg):
    app = create_app(cfg, upstream_transport=FakeUpstream().transport())
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7300")


@pytest.fixture
async def client(cfg):
    app, c = await client_for(cfg)
    async with app.router.lifespan_context(app), c:
        yield c


@pytest.fixture
async def wide_client(cfg):
    app, c = await client_for(replace(cfg, host="0.0.0.0"))
    async with app.router.lifespan_context(app), c:
        yield c


@pytest.fixture
def tokens(cfg):
    return load_tokens(cfg.data_dir)


def test_tokens_are_made_once_and_differ(cfg):
    first, second = load_tokens(cfg.data_dir), load_tokens(cfg.data_dir)
    assert first == second and first.owner != first.agent and len(first.owner) > 30


async def test_foreign_host_is_refused_everywhere(client):
    for path in ("/a/bot/v1/models", "/api/v1/models", "/"):
        assert (await client.get(path, headers={"Host": "evil.example"})).status_code == 403


async def test_allowed_hosts_are_let_in(cfg, tokens):
    app, c = await client_for(replace(cfg, allowed_hosts=("box.example",)))
    async with app.router.lifespan_context(app), c:
        resp = await c.get("/api/v1/status", headers={"Host": "box.example:7300", **bearer(tokens.owner)})
    assert resp.status_code == 200


async def test_text_plain_post_is_refused(client):
    resp = await client.post(URL, content=b"{}", headers={"Content-Type": "text/plain"})
    assert resp.status_code == 415


async def test_no_cors_headers(client):
    resp = await client.options("/api/v1/models", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in resp.headers


async def test_api_needs_a_token(client, tokens):
    assert (await client.get("/api/v1/models")).status_code == 401
    assert (await client.get("/api/v1/models", headers=bearer("wrong"))).status_code == 401
    assert (await client.get("/api/v1/models", headers=bearer(tokens.agent))).status_code == 200


async def test_proxy_needs_no_token_on_loopback_but_does_beyond_it(client, wide_client, tokens):
    assert (await client.get("/a/bot/v1/models")).status_code == 200
    assert (await wide_client.get("/a/bot/v1/models")).status_code == 401
    assert (await wide_client.get("/a/bot/v1/models", headers=bearer(tokens.agent))).status_code == 200
