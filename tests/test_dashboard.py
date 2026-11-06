from pathlib import Path

import httpx
import pytest
from fakes import FakeUpstream

import provingbench
from provingbench.app import create_app
from provingbench.auth import load_tokens

STATIC = Path(provingbench.__file__).parent / "static"


@pytest.fixture
async def app(cfg):
    app = create_app(cfg, upstream_transport=FakeUpstream().transport())
    async with app.router.lifespan_context(app):
        yield app


def http(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7300")


@pytest.fixture
async def client(app):
    async with http(app) as c:
        yield c


@pytest.fixture
def tokens(cfg):
    return load_tokens(cfg.data_dir)


async def test_opening_with_the_token_sets_a_strict_cookie_and_drops_it_from_the_url(client, tokens):
    resp = await client.get(f"/?token={tokens.owner}", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    cookie = resp.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie.replace("Strict", "strict")


async def test_a_wrong_token_gets_no_cookie(client):
    resp = await client.get("/?token=nope", follow_redirects=False)
    assert "set-cookie" not in resp.headers and resp.status_code == 200


async def test_the_agent_token_gets_no_cookie(client, tokens):
    assert "set-cookie" not in (await client.get(f"/?token={tokens.agent}", follow_redirects=False)).headers


async def test_the_cookie_works_on_the_api(client, tokens):
    await client.get(f"/?token={tokens.owner}", follow_redirects=False)
    assert (await client.get("/api/v1/agents")).status_code == 200


async def test_without_the_cookie_the_api_stays_shut(client):
    assert (await client.get("/api/v1/agents")).status_code == 401


async def test_pages_send_a_strict_csp(client):
    resp = await client.get("/")
    assert "default-src 'self'" in resp.headers["content-security-policy"]
    assert "<title>ProvingBench</title>" in resp.text


async def test_static_files_are_served(client):
    assert (await client.get("/static/app.js")).status_code == 200
    assert (await client.get("/static/style.css")).status_code == 200


def test_app_js_never_builds_html_from_strings():
    text = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in text and "insertAdjacentHTML" not in text and "document.write" not in text


async def test_models_say_which_are_cloud(app, client, tokens):
    await client.get(f"/?token={tokens.owner}", follow_redirects=False)
    rows = (await client.get("/api/v1/models")).json()["models"]
    assert rows[0]["cloud"] is False


def test_every_view_is_registered():
    text = (STATIC / "app.js").read_text(encoding="utf-8")
    for view in ("scoreboard", "run", "result", "model", "live", "models"):
        assert f"{view}:" in text or f"\"{view}\"" in text
    assert "scoreboardView" in text and "resultView" in text
