"""Who may ask for what. Everything is refused from a foreign Host, so a web page can't reach ProvingBench."""
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlsplit

COOKIE = "provingbench_owner"
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class Tokens:
    owner: str
    agent: str


def _token(path: Path) -> str:
    try:
        if text := path.read_text(encoding="utf-8").strip():
            return text
    except FileNotFoundError:
        pass
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    return token


def load_tokens(data_dir: Path) -> Tokens:
    data_dir.mkdir(parents=True, exist_ok=True)
    return Tokens(_token(data_dir / "owner_token"), _token(data_dir / "agent_token"))


def _hostname(value: str) -> str | None:
    return urlsplit(f"//{value}").hostname


class Guard:
    def __init__(self, app, cfg, tokens: Tokens):
        self.app, self.tokens = app, tokens
        self.hosts = _LOOPBACK | set(cfg.allowed_hosts)
        self.wide = cfg.host not in _LOOPBACK

    def _role(self, headers: dict[bytes, bytes]) -> str | None:
        kind, _, value = headers.get(b"authorization", b"").decode("latin-1").partition(" ")
        if kind.lower() == "bearer":
            if hmac.compare_digest(value.encode(), self.tokens.owner.encode()):
                return "owner"
            if hmac.compare_digest(value.encode(), self.tokens.agent.encode()):
                return "agent"
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get(b"cookie", b"").decode("latin-1"))
        except Exception:
            return None
        if COOKIE in cookie and hmac.compare_digest(cookie[COOKIE].value.encode(), self.tokens.owner.encode()):
            return "owner"
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        path, method = scope["path"], scope["method"]
        if _hostname(headers.get(b"host", b"").decode("latin-1")) not in self.hosts:
            return await _refuse(send, 403, "forbidden", "this Host isn't allowed")
        is_api, is_proxy = path.startswith("/api/"), path.startswith("/a/")
        if (is_api or is_proxy) and method in ("POST", "PUT") and \
                not headers.get(b"content-type", b"").lower().startswith(b"application/json"):
            return await _refuse(send, 415, "unsupported_media_type", "the body must be application/json")
        role = self._role(headers)
        if role is None and (is_api or (is_proxy and self.wide)):
            return await _refuse(send, 401, "unauthorized", "a token is needed")
        scope.setdefault("state", {})["role"] = role
        return await self.app(scope, receive, send)


async def _refuse(send, status: int, code: str, message: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
