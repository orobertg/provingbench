"""Targets: how to hand an agent a message and read its reply, written down so any HTTP agent connects without code."""
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import httpx

from provingbench.proxy import valid_agent

_PLACEHOLDER = re.compile(r"\{(session|message)\}")
DEFAULT_TIMEOUT_S = 300


class TargetError(Exception):
    pass


@dataclass(frozen=True)
class Target:
    id: str
    agent_id: str
    url: str
    token_env: str
    body: dict
    headers: dict
    reply_path: str
    timeout_s: float


@dataclass(frozen=True)
class TriggerReply:
    reply: str | None = None
    error: str | None = None
    unreachable: bool = False


def _mentions_message(value) -> bool:
    if isinstance(value, str):
        return "{message}" in value
    if isinstance(value, dict):
        return any(map(_mentions_message, value.values()))
    return isinstance(value, list) and any(map(_mentions_message, value))


def _target(data: dict) -> Target:
    for key in ("id", "agent_id", "url", "reply_path"):
        if not (isinstance(data.get(key), str) and data[key]):
            raise TargetError(f"{key} must be some text")
    if not valid_agent(data["agent_id"]):
        raise TargetError("agent_id can use only letters, numbers, dots, dashes and underscores")
    if not data["url"].startswith(("http://", "https://")):
        raise TargetError("url must start with http:// or https://")
    if not (isinstance(data.get("body"), dict) and _mentions_message(data["body"])):
        raise TargetError("body must include {message} somewhere")
    headers = data.get("headers", {})
    if not isinstance(headers, dict):
        raise TargetError("headers must be a table")
    return Target(data["id"], data["agent_id"], data["url"], data.get("token_env", ""), data["body"], headers,
                  data["reply_path"], data.get("timeout_s", DEFAULT_TIMEOUT_S))


def load_targets(folder: Path) -> dict[str, Target]:
    targets: dict[str, Target] = {}
    for file in sorted(folder.glob("*.toml")) if folder.is_dir() else ():
        try:
            target = _target(tomllib.loads(file.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, TargetError) as e:
            raise TargetError(f"{file}: {e}") from e
        if target.id in targets:
            raise TargetError(f"{file}: the id {target.id!r} is already used by another target")
        targets[target.id] = target
    return targets


def _filled(value, values: dict[str, str]):
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: values[m.group(1)], value)
    if isinstance(value, dict):
        return {k: _filled(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [_filled(v, values) for v in value]
    return value


def _at_path(data, path: str):
    for part in path.split("."):
        if isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        elif isinstance(data, dict) and part in data:
            data = data[part]
        else:
            return None
    return data


async def send(target: Target, client: httpx.AsyncClient, session: str, message: str) -> TriggerReply:
    values = {"session": session, "message": message}
    headers = _filled(target.headers, values)
    if target.token_env and (token := os.environ.get(target.token_env)):
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = await client.post(target.url, json=_filled(target.body, values), headers=headers, timeout=target.timeout_s)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        return TriggerReply(error=f"couldn't reach {target.url}: {e}", unreachable=True)
    except httpx.HTTPError as e:
        return TriggerReply(error=f"{type(e).__name__} from {target.url}: {e}")
    if resp.status_code >= 400:
        return TriggerReply(error=f"HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        reply = _at_path(resp.json(), target.reply_path)
    except ValueError:
        return TriggerReply(error=f"the answer wasn't JSON: {resp.text[:300]}")
    if not isinstance(reply, str):
        return TriggerReply(error=f"no reply text at {target.reply_path}: {resp.text[:300]}")
    return TriggerReply(reply=reply)
