from dataclasses import replace

import pytest

from provingbench.config import load_config
from provingbench.store import CallRecord, Store


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test gets its own data folder and no real keys."""
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("PROVINGBENCH_HOME", str(folder))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return folder


@pytest.fixture
def cfg(home):
    return load_config()


@pytest.fixture
def store(cfg):
    s = Store(cfg.data_dir / "t.db")
    yield s
    s.close()


def record(**over) -> CallRecord:
    base = CallRecord(agent_id="a", model="ollama/m", upstream="ollama", state="ok", status=200,
                      request={"messages": []}, response={"text": "hi"}, tool_calls=[], finish="stop",
                      ttft_s=0.1, total_s=0.2, prompt_tokens=1, completion_tokens=1, cost_usd=0.0, flags=[])
    return replace(base, **over)
