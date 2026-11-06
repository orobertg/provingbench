"""Settings: a TOML file in ProvingBench's home folder, every value with a default."""
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    def __init__(self, field: str, problem: str):
        super().__init__(f"{field}: {problem}")
        self.field = field


@dataclass(frozen=True)
class Upstream:
    name: str
    base_url: str
    api_key_env: str = ""


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    allowed_hosts: tuple[str, ...]
    data_dir: Path
    upstreams: dict[str, Upstream]
    default_upstream: str
    slow_first_token_s: float
    retention_days: int | None
    judge_model: str
    max_cost_usd: float
    repeats: int
    suites: dict[str, Path]
    ollama_logprobs: bool


def home() -> Path:
    if folder := os.environ.get("PROVINGBENCH_HOME"):
        return Path(folder)
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "provingbench"
    return Path.home() / ".config" / "provingbench"


DEFAULT_UPSTREAMS = {"ollama": Upstream("ollama", "http://localhost:11434/v1"),
                     "openrouter": Upstream("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")}


def _read(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as e:
        raise ConfigError(str(path), f"can't read it: {e}") from e


def _upstreams(section: dict) -> dict[str, Upstream]:
    found = dict(DEFAULT_UPSTREAMS)
    for name, value in section.items():
        if name == "default":
            continue
        if not isinstance(value, dict) or not str(value.get("base_url", "")).startswith(("http://", "https://")):
            raise ConfigError(f"upstreams.{name}.base_url", "must start with http:// or https://")
        found[name] = Upstream(name, value["base_url"], value.get("api_key_env", ""))
    return found


def load_config(path: Path | None = None) -> Config:
    folder = home()
    raw = _read(path or folder / "config.toml")
    server, runs, section = raw.get("server", {}), raw.get("runs", {}), raw.get("upstreams", {})
    port = server.get("port", 7300)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("server.port", "must be between 1 and 65535")
    upstreams = _upstreams(section)
    default = section.get("default", "ollama")
    if default not in upstreams:
        raise ConfigError("upstreams.default", f"no upstream named {default!r}")
    log = raw.get("log", {})
    return Config(host=server.get("host", "127.0.0.1"), port=port, allowed_hosts=tuple(server.get("allowed_hosts", ())),
                  data_dir=folder, upstreams=upstreams, default_upstream=default,
                  slow_first_token_s=log.get("slow_first_token_s", 30), retention_days=log.get("retention_days"),
                  judge_model=runs.get("judge_model", ""), max_cost_usd=runs.get("max_cost_usd", 1.0),
                  repeats=runs.get("repeats", 3), suites={k: Path(v) for k, v in raw.get("suites", {}).items()},
                  ollama_logprobs=log.get("ollama_logprobs", True))
