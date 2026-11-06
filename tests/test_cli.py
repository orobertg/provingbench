import socket

import httpx
import pytest

from provingbench import cli
from provingbench.auth import load_tokens
from provingbench.cli import main


def answers(monkeypatch, *replies):
    queue = list(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": queue.pop(0))


@pytest.fixture
def ollama_up(monkeypatch):
    monkeypatch.setattr(cli, "_reachable", lambda url: True)


@pytest.fixture
def ollama_down(monkeypatch):
    monkeypatch.setattr(cli, "_reachable", lambda url: False)


def test_setup_writes_a_config_with_the_key_variable_name_and_no_key(home, ollama_up, monkeypatch, capsys):
    answers(monkeypatch, "MY_ROUTER_KEY")
    assert main(["setup"]) == 0
    text = (home / "config.toml").read_text()
    assert "MY_ROUTER_KEY" in text and "sk-" not in text
    assert cli.load_config().upstreams["openrouter"].api_key_env == "MY_ROUTER_KEY"
    assert "ollama: ok" in capsys.readouterr().out


def test_setup_with_a_blank_answer_keeps_the_usual_name(home, ollama_up, monkeypatch):
    answers(monkeypatch, "")
    main(["setup"])
    assert cli.load_config().upstreams["openrouter"].api_key_env == "OPENROUTER_API_KEY"


def test_setup_never_overwrites_a_config(home, ollama_up, monkeypatch, capsys):
    (home / "config.toml").write_text("# mine\n")
    answers(monkeypatch)
    assert main(["setup"]) == 0
    assert (home / "config.toml").read_text() == "# mine\n" and "already" in capsys.readouterr().out


def test_doctor_reports_each_upstream(ollama_up, capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ollama: ok" in out and "openrouter: no key" in out


def test_doctor_fails_when_a_local_upstream_is_down(ollama_down, capsys):
    assert main(["doctor"]) == 1
    assert "ollama: not reachable" in capsys.readouterr().out


def test_doctor_counts_a_key_as_configured(ollama_up, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    main(["doctor"])
    out = capsys.readouterr().out
    assert "openrouter: key set" in out and "secret-value" not in out


def test_use_without_a_server_says_so(capsys, monkeypatch):
    def refused(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "request", refused)
    assert main(["use", "bot", "ollama/m"]) == 1
    assert "provingbench serve" in capsys.readouterr().err


def test_use_sends_the_owner_token(cfg, capsys, monkeypatch):
    seen = {}

    def fake(method, url, **kw):
        seen.update(method=method, url=url, **kw)
        return httpx.Response(200, json={"model": "ollama/m"})
    monkeypatch.setattr(httpx, "request", fake)
    assert main(["use", "bot", "ollama/m"]) == 0
    assert seen["method"] == "PUT" and seen["url"].endswith("/api/v1/agents/bot/model")
    assert seen["headers"]["Authorization"] == f"Bearer {load_tokens(cfg.data_dir).owner}"
    assert seen["json"] == {"model": "ollama/m"} and "ollama/m" in capsys.readouterr().out


def test_an_api_refusal_is_shown_plainly(capsys, monkeypatch):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: httpx.Response(
        403, json={"error": {"code": "forbidden", "message": "not on the list"}}))
    assert main(["use", "bot", "x"]) == 1
    assert "not on the list" in capsys.readouterr().err


def test_models_lists_ids_and_prices(capsys, monkeypatch):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: httpx.Response(200, json={"models": [
        {"id": "ollama/m", "price_prompt": 0, "price_completion": 0, "supports_tools": None},
        {"id": "openrouter/x/y", "price_prompt": 1e-6, "price_completion": 2e-6, "supports_tools": True}]}))
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "ollama/m" in out and "free" in out and "openrouter/x/y" in out and "$1.00" in out


def test_dashboard_prints_the_sign_in_address(cfg, capsys):
    assert main(["dashboard"]) == 0
    assert f"http://127.0.0.1:7300/?token={load_tokens(cfg.data_dir).owner}" in capsys.readouterr().out


def test_run_starts_follows_and_summarizes(capsys, monkeypatch):
    sent, answers_left = [], [
        httpx.Response(200, json={"estimate_usd": 0.25}),
        httpx.Response(202, json={"id": 7}),
        httpx.Response(200, json={"state": "running", "total": 2, "results": [], "spent_usd": 0}),
        httpx.Response(200, json={"state": "done", "total": 2, "spent_usd": 0.5, "error": None, "results": [
            {"model": "ollama/m", "outcome": "pass"}, {"model": "ollama/m", "outcome": "fail"}]})]

    def fake(method, url, **kw):
        sent.append((method, url, kw.get("json")))
        return answers_left.pop(0)
    monkeypatch.setattr(httpx, "request", fake)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    code = main(["run", "--target", "t", "--suite", "s", "--model", "ollama/m", "--repeats", "2", "--label", "v1"])
    assert code == 0 and sent[1][2] == {"target": "t", "suite": "s", "models": ["ollama/m"], "repeats": 2, "label": "v1"}
    out = capsys.readouterr().out
    assert "ollama/m: 1 of 2 passed" in out and "done" in out and "$0.25" in out


def test_run_reports_a_failed_run(capsys, monkeypatch):
    answers_left = [httpx.Response(200, json={"estimate_usd": None}), httpx.Response(202, json={"id": 7}), httpx.Response(200, json={
        "state": "failed", "total": 1, "spent_usd": 0, "error": "couldn't reach it", "results": []})]
    monkeypatch.setattr(httpx, "request", lambda *a, **k: answers_left.pop(0))
    assert main(["run", "--target", "t", "--suite", "s", "--model", "m"]) == 1
    captured = capsys.readouterr()
    assert "couldn't reach it" in captured.err and "unknown" in captured.out


def test_results_prints_the_board(capsys, monkeypatch):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: httpx.Response(200, json={
        "tasks": [{"id": "t", "tags": []}], "models": ["m"],
        "cells": [{"model": "m", "task_id": "t", "passed": 2, "failed": 1, "errors": 0, "rate": 2 / 3, "stale": False}]}))
    assert main(["results"]) == 0
    out = capsys.readouterr().out
    assert "m" in out and "t" in out and "2/3" in out


def test_results_with_nothing_yet_says_so(capsys, monkeypatch):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: httpx.Response(200, json={"tasks": [], "models": [], "cells": []}))
    assert main(["results"]) == 0
    assert "No results yet" in capsys.readouterr().out


def test_serve_on_a_busy_port_says_so(capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert main(["serve", "--port", str(port)]) == 1
    assert str(port) in capsys.readouterr().err


def test_a_bad_config_is_one_clean_line(home, capsys):
    (home / "config.toml").write_text("[server]\nport = 0\n")
    assert main(["doctor"]) == 1
    assert "server.port" in capsys.readouterr().err
