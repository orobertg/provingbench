"""Where model calls go: routing a model ID to its upstream, and what each upstream offers."""
import os
import sqlite3
import time
from urllib.parse import urlsplit

import httpx

from provingbench.config import Config, Upstream

LISTING_MAX_AGE_S = 3600
MAX_PAGES = 20
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def is_local(upstream: Upstream) -> bool:
    return urlsplit(upstream.base_url).hostname in _LOOPBACK


def route(cfg: Config, model_id: str) -> tuple[Upstream, str]:
    """`openrouter/x/y` goes to openrouter as `x/y`; a name with no known prefix goes whole to the default."""
    prefix, _, rest = model_id.partition("/")
    if rest and prefix in cfg.upstreams:
        return cfg.upstreams[prefix], rest
    return cfg.upstreams[cfg.default_upstream], model_id


def _price(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row(upstream: Upstream, item: dict) -> dict:
    pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
    params = item.get("supported_parameters")
    local = is_local(upstream)
    return {"id": f"{upstream.name}/{item['id']}", "upstream": upstream.name,
            "price_prompt": 0 if local else _price(pricing.get("prompt")),
            "price_completion": 0 if local else _price(pricing.get("completion")),
            "context_length": item.get("context_length"),
            "supports_tools": "tools" in params if isinstance(params, list) else None}


class Upstreams:
    def __init__(self, cfg: Config, store, transport=None):
        self.cfg, self.store, self._transport = cfg, store, transport
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._refreshed_at: float | None = None

    def client(self, upstream: Upstream) -> httpx.AsyncClient:
        if upstream.name not in self._clients:
            key = os.environ.get(upstream.api_key_env) if upstream.api_key_env else None
            self._clients[upstream.name] = httpx.AsyncClient(
                base_url=upstream.base_url, transport=self._transport, trust_env=not is_local(upstream),
                timeout=httpx.Timeout(300.0, connect=10.0), headers={"Authorization": f"Bearer {key}"} if key else {})
        return self._clients[upstream.name]

    async def _fetch(self, upstream: Upstream) -> list[dict]:
        rows, url = [], "/models"
        for _ in range(MAX_PAGES):
            resp = await self.client(upstream).get(url)
            resp.raise_for_status()
            body = resp.json()
            rows += [_row(upstream, item) for item in body.get("data", []) if isinstance(item.get("id"), str)]
            url = (body.get("links") or {}).get("next")
            if not url:
                break
        return rows

    async def refresh(self) -> dict[str, str]:
        """Update the saved list from each upstream; one failing keeps its old rows and is reported."""
        errors = {}
        for upstream in self.cfg.upstreams.values():
            if upstream.api_key_env and not os.environ.get(upstream.api_key_env):
                continue
            try:
                self.store.save_models(await self._fetch(upstream))
            except (httpx.HTTPError, ValueError, AttributeError, TypeError, sqlite3.Error) as e:
                errors[upstream.name] = str(e) or type(e).__name__
        self._refreshed_at = time.monotonic()
        return errors

    async def listing(self) -> list[dict]:
        if self._refreshed_at is None or time.monotonic() - self._refreshed_at > LISTING_MAX_AGE_S:
            await self.refresh()
        return self.store.models()

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()


def is_paid(cfg: Config, store, model_id: str) -> bool:
    """Local models and listed free ones aren't paid; an unknown price counts as paid."""
    upstream, _ = route(cfg, model_id)
    if is_local(upstream):
        return False
    row = store.model(model_id)
    return not row or row["price_prompt"] != 0 or row["price_completion"] != 0


def cost_of(cfg: Config, store, model_id: str, usage: dict | None) -> float | None:
    """What a call cost: the upstream's own figure if given, else tokens times the listed price."""
    upstream, _ = route(cfg, model_id)
    if is_local(upstream):
        return 0
    if not usage:
        return None
    if isinstance(usage.get("cost"), int | float):
        return usage["cost"]
    row = store.model(model_id)
    if not row or row["price_prompt"] is None or row["price_completion"] is None:
        return None
    return (usage.get("prompt_tokens") or 0) * row["price_prompt"] + \
        (usage.get("completion_tokens") or 0) * row["price_completion"]
