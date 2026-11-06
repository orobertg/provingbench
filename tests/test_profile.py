import httpx
import pytest
from conftest import record
from fakes import FakeUpstream

from provingbench.app import create_app
from provingbench.auth import load_tokens
from provingbench.profile import profile, scoreboard
from provingbench.store import Flag


def add_results(store, model, task, task_hash, outcomes, tags=()):
    """One run's results; returns their IDs."""
    run_id = store.create_run("t", "bot", "s", [model], 1, 1.0, "", None, len(outcomes))
    return [store.add_result(run_id, model, task, task_hash, list(tags), i, f"a{run_id}-{i}", outcome, [], "r", None, 1.0, 0.01)
            for i, outcome in enumerate(outcomes)]


def only_cell(board):
    (cell,) = board["cells"]
    return cell


def cell_for(board, model):
    return next(c for c in board["cells"] if c["model"] == model)


def test_cell_rate_ignores_errors(store):
    add_results(store, "m", "t", "h1", ["pass", "fail", "error"])
    cell = only_cell(scoreboard(store))
    assert (cell["passed"], cell["failed"], cell["errors"]) == (1, 1, 1) and cell["rate"] == 0.5


def test_only_errors_is_no_rate_not_zero(store):
    add_results(store, "m", "t", "h1", ["error"])
    assert only_cell(scoreboard(store))["rate"] is None


def test_scores_compare_only_within_the_latest_task_hash(store):
    add_results(store, "m", "t", "old", ["fail", "fail"])
    add_results(store, "m", "t", "new", ["pass"])
    assert only_cell(scoreboard(store))["rate"] == 1.0


def test_a_later_run_replaces_an_earlier_one_with_the_same_hash(store):
    add_results(store, "m", "t", "h", ["fail"])
    add_results(store, "m", "t", "h", ["pass"])
    assert only_cell(scoreboard(store))["rate"] == 1.0


def test_model_only_tested_on_an_old_hash_shows_stale(store):
    add_results(store, "a", "t", "old", ["pass"])
    add_results(store, "b", "t", "new", ["pass"])
    board = scoreboard(store)
    assert cell_for(board, "a")["stale"] and not cell_for(board, "b")["stale"]


def test_owner_override_changes_the_rate_but_not_the_grade(store):
    rid = add_results(store, "m", "t", "h", ["fail"])[0]
    store.add_override(rid, "pass", "the model was right")
    assert only_cell(scoreboard(store))["rate"] == 1.0 and store.result(rid)["outcome"] == "fail"


def test_the_board_lists_models_and_tasks(store):
    add_results(store, "m", "t", "h", ["pass"], tags=["chat"])
    board = scoreboard(store)
    assert board["models"] == ["m"] and board["tasks"] == [{"id": "t", "tags": ["chat"]}]


def test_an_empty_board(store):
    assert scoreboard(store) == {"tasks": [], "models": [], "cells": []}


def seed_two_tags_one_flagged(store):
    add_results(store, "m", "status", "h", ["pass", "fail"], tags=["tools"])
    add_results(store, "m", "hello", "h", ["pass"], tags=["chat"])
    store.add_call(record(model="m", flags=[Flag("empty_reply")], ttft_s=2.0))
    store.add_call(record(model="m", ttft_s=4.0))
    store.add_call(record(model="other", flags=[Flag("slow")], ttft_s=99.0))


def test_profile_rates_by_tag_and_counts_flags(store):
    seed_two_tags_one_flagged(store)
    p = profile(store, "m")
    assert p["by_tag"] == {"tools": 0.5, "chat": 1.0} and p["flags"] == [{"name": "empty_reply", "count": 1}]
    assert p["failed_tasks"] == ["status"] and p["median_ttft_s"] == 3.0
    assert p["cost_per_task"] == pytest.approx(0.01) and p["last_run"] and p["model"] == "m"


def test_profile_of_an_untested_model_is_empty_not_an_error(store):
    p = profile(store, "ollama/never")
    assert p["by_tag"] == {} and p["last_run"] is None and p["median_ttft_s"] is None and p["notes"] == ""


async def test_the_api_serves_board_and_profile_with_slashes_in_the_model(cfg):
    app = create_app(cfg, upstream_transport=FakeUpstream().transport())
    head = {"Authorization": f"Bearer {load_tokens(cfg.data_dir).agent}"}
    async with app.router.lifespan_context(app):
        add_results(app.state.store, "openrouter/x/y", "t", "h", ["pass"], tags=["chat"])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7300") as c:
            assert (await c.get("/api/v1/scoreboard", headers=head)).json()["models"] == ["openrouter/x/y"]
            body = (await c.get("/api/v1/models/openrouter/x/y/profile", headers=head)).json()
            assert body["by_tag"] == {"chat": 1.0}
