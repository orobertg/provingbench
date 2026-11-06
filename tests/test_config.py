import pytest

from provingbench.config import ConfigError, home, load_config


def test_defaults_without_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVINGBENCH_HOME", str(tmp_path))
    cfg = load_config()
    assert (cfg.host, cfg.port) == ("127.0.0.1", 7300)
    assert cfg.upstreams["ollama"].base_url == "http://localhost:11434/v1"
    assert cfg.upstreams["openrouter"].api_key_env == "OPENROUTER_API_KEY"
    assert cfg.default_upstream == "ollama" and cfg.max_cost_usd == 1.0 and cfg.repeats == 3
    assert cfg.slow_first_token_s == 30 and cfg.retention_days is None and cfg.ollama_logprobs


def test_bad_port_names_the_field(tmp_path):
    (tmp_path / "c.toml").write_text("[server]\nport = 99999\n")
    with pytest.raises(ConfigError, match="server.port"):
        load_config(tmp_path / "c.toml")


def test_bad_toml_names_the_file(tmp_path):
    (tmp_path / "c.toml").write_text("[server\n")
    with pytest.raises(ConfigError, match="c.toml"):
        load_config(tmp_path / "c.toml")


def test_file_overrides_defaults_and_names_suites(tmp_path):
    (tmp_path / "c.toml").write_text(
        '[server]\nport = 7400\n[suites]\nmine = "/tasks/mine"\n'
        '[upstreams.other]\nbase_url = "http://x/v1"\napi_key_env = "OTHER_KEY"\n[runs]\nmax_cost_usd = 2.5\n')
    cfg = load_config(tmp_path / "c.toml")
    assert cfg.port == 7400 and cfg.suites["mine"].as_posix().endswith("/tasks/mine") and cfg.max_cost_usd == 2.5
    assert cfg.upstreams["other"].api_key_env == "OTHER_KEY" and "ollama" in cfg.upstreams


def test_unknown_default_upstream_is_refused(tmp_path):
    (tmp_path / "c.toml").write_text('[upstreams]\ndefault = "nope"\n')
    with pytest.raises(ConfigError, match="upstreams.default"):
        load_config(tmp_path / "c.toml")


def test_home_follows_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVINGBENCH_HOME", str(tmp_path / "x"))
    assert home() == tmp_path / "x"
