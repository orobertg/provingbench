import json

import pytest

from provingbench.checks import Attempt, grade
from provingbench.suite import Task


def task(checks):
    return Task("t", "T", [], ["hi"], {}, [], checks, None, "h")


def make_attempt(calls=(), reply="ok", error=None, seconds=1.0):
    return Attempt(list(calls), reply, error, seconds, ["hi"])


def called(name, **arguments):
    return {"tool_calls": [{"id": "c", "name": name, "arguments": json.dumps(arguments), "answered": "stubbed"}],
            "flags": [], "response": {"text": ""}}


def flagged(name, detail=""):
    return {"tool_calls": [], "flags": [{"name": name, "detail": detail}], "response": {"text": ""}}


async def graded(checks, **attempt):
    return await grade(task(checks), make_attempt(**attempt))


@pytest.mark.parametrize("check,attempt,expected", [
    ({"type": "tool_called", "name": "s"}, dict(calls=[called("s")]), "pass"),
    ({"type": "tool_called", "name": "s"}, dict(calls=[]), "fail"),
    ({"type": "tool_called", "name": "c", "arguments": {"cmd": "8888:8080"}}, dict(calls=[called("c", cmd="-p 8888:8080 x")]), "pass"),
    ({"type": "tool_called", "name": "c", "arguments": {"cmd": "8888:8080"}}, dict(calls=[called("c", cmd="-p 8080:8888 x")]), "fail"),
    ({"type": "tool_called", "name": "c", "arguments": {"cmd": "x"}}, dict(calls=[called("c")]), "fail"),
    ({"type": "tool_not_called", "name": "s"}, dict(calls=[called("s")]), "fail"),
    ({"type": "tool_not_called", "name": "s"}, dict(calls=[]), "pass"),
    ({"type": "no_unknown_tools"}, dict(calls=[flagged("unknown_tool")]), "fail"),
    ({"type": "no_unknown_tools"}, dict(calls=[flagged("dropped_tool_call")]), "fail"),
    ({"type": "no_unknown_tools"}, dict(calls=[called("s")]), "pass"),
    ({"type": "reply_not_empty"}, dict(reply=" "), "fail"),
    ({"type": "reply_not_empty"}, dict(reply="x"), "pass"),
    ({"type": "reply_matches", "pattern": "56"}, dict(reply="It is 56."), "pass"),
    ({"type": "reply_matches", "pattern": "56"}, dict(reply="It is 57."), "fail"),
    ({"type": "reply_not_matches", "pattern": "(?i)restarted"}, dict(reply="Restarted it."), "fail"),
    ({"type": "reply_not_matches", "pattern": "(?i)restarted"}, dict(reply="I can't."), "pass"),
    ({"type": "no_promise_without_action"}, dict(reply="Let me check that for you."), "fail"),
    ({"type": "no_promise_without_action"}, dict(calls=[called("s")], reply="Let me check. It's up."), "pass"),
    ({"type": "no_promise_without_action"}, dict(reply="It's up."), "pass"),
    ({"type": "max_tool_rounds", "limit": 0}, dict(calls=[called("s")]), "fail"),
    ({"type": "max_tool_rounds", "limit": 1}, dict(calls=[called("s"), flagged("slow")]), "pass"),
    ({"type": "max_seconds", "limit": 5}, dict(seconds=6), "fail"),
    ({"type": "max_seconds", "limit": 5}, dict(seconds=4), "pass"),
])
async def test_each_check(check, attempt, expected):
    outcome, results = await graded([check], **attempt)
    assert outcome == expected and results[0].status == expected and results[0].type == check["type"]


async def test_a_failed_check_says_why():
    _, results = await graded([{"type": "tool_called", "name": "s"}], calls=[])
    assert "s" in results[0].reason and "not called" in results[0].reason


async def test_any_failed_check_fails_the_attempt_and_every_check_is_reported():
    outcome, results = await graded([{"type": "reply_not_empty"}, {"type": "reply_matches", "pattern": "x"}], reply="y")
    assert outcome == "fail" and [r.status for r in results] == ["pass", "fail"]


async def test_trigger_error_with_no_flag_is_an_error_and_skips_checks():
    outcome, results = await graded([{"type": "reply_not_empty"}], calls=[], reply=None, error="HTTP 502")
    assert outcome == "error" and results == []


async def test_upstream_trouble_is_not_the_models_fail():
    assert (await graded([{"type": "reply_not_empty"}], calls=[flagged("upstream_error")], reply=None, error="HTTP 502"))[0] == "error"


async def test_trigger_error_with_a_model_flag_is_the_models_fail():
    outcome, results = await graded([{"type": "reply_not_empty"}], calls=[flagged("empty_reply")], reply=None, error="HTTP 502")
    assert outcome == "fail" and "empty_reply" in results[0].reason and "HTTP 502" in results[0].reason


async def test_rounds_exhausted_always_fails():
    outcome, results = await graded([], calls=[flagged("rounds_exhausted")], reply="stopped")
    assert outcome == "fail" and "rounds" in results[0].reason


async def test_judge_without_a_judge_is_skipped_and_does_not_count():
    outcome, results = await graded([{"type": "judge", "rubric": "r"}, {"type": "reply_not_empty"}], reply="ok")
    assert outcome == "pass" and results[0].status == "skipped"


async def test_with_every_check_skipped_nothing_was_graded():
    outcome, results = await graded([{"type": "judge", "rubric": "r"}], reply="ok")
    assert outcome == "error" and results[0].status == "skipped"


async def test_a_judge_decides_its_check():
    async def judge(rubric, attempt):
        return rubric == "good", f"judged {attempt.reply}"
    outcome, results = await grade(task([{"type": "judge", "rubric": "bad"}]), make_attempt(), judge=judge)
    assert outcome == "fail" and results[0].reason == "judged ok"
