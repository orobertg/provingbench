import pytest

from provingbench.flags import detect, promises_action
from provingbench.stream import Completion


def tool_schema(name, required=(), properties=None):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": properties or {}, "required": list(required)}}}


def req(*tools):
    return {"messages": [], "tools": list(tools)} if tools else {"messages": []}


def offered(*names):
    return req(*[tool_schema(n) for n in names])


def call(name, arguments):
    return Completion(tool_calls=[{"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}],
                      finish="tool_calls")


def names(flags):
    return {f.name for f in flags}


@pytest.mark.parametrize("c,tools,expected", [
    (Completion(finish="stop"), [], {"empty_reply"}),
    (Completion(text="  \n", finish="stop"), [], {"empty_reply"}),
    (Completion(text="Let me check the container status for you. One moment please!"), ["system_report"],
     {"promised_no_action"}),
    (Completion(text="The container is running."), ["system_report"], set()),
    (Completion(text="Let me check."), [], set()),      # no tools were offered
    (call("docker_ps", "{}"), ["system_report"], {"unknown_tool"}),
    (call("system_report", "{bad"), ["system_report"], {"bad_arguments"}),
    (call("system_report", "[1]"), ["system_report"], {"bad_arguments"}),
    (call("system_report", ""), ["system_report"], set()),
    (call("system_report", "{}"), ["system_report"], set()),
])
def test_flags(c, tools, expected):
    assert names(detect(offered(*tools), c, ttft_s=1, slow_s=30)) == expected


def test_a_promise_alongside_a_call_is_fine():
    c = call("system_report", "{}")
    c.text = "Let me check."
    assert names(detect(offered("system_report"), c, ttft_s=1, slow_s=30)) == set()


def test_unknown_tool_names_it():
    flags = detect(offered("a"), call("docker_ps", "{}"), ttft_s=1, slow_s=30)
    assert flags[0].detail == "docker_ps"


def test_missing_required_argument_and_wrong_type_are_bad():
    tool = tool_schema("t", required=["container"], properties={"lines": {"type": "integer"}})
    assert names(detect(req(tool), call("t", '{"lines": 5}'), ttft_s=1, slow_s=30)) == {"bad_arguments"}
    assert names(detect(req(tool), call("t", '{"container": "c", "lines": "5"}'), ttft_s=1, slow_s=30)) == {"bad_arguments"}
    assert names(detect(req(tool), call("t", '{"container": "c", "lines": 5}'), ttft_s=1, slow_s=30)) == set()


def test_true_is_not_an_integer():
    tool = tool_schema("t", properties={"n": {"type": "integer"}})
    assert names(detect(req(tool), call("t", '{"n": true}'), ttft_s=1, slow_s=30)) == {"bad_arguments"}


def test_dropped_call_is_named_from_logprob_tokens():
    c = Completion(finish="stop", raw_tokens='<tool_call>{"name": "docker_ps", "arguments": {}}</tool_call>')
    flags = {f.name: f.detail for f in detect(req(), c, ttft_s=1, slow_s=30)}
    assert flags["dropped_tool_call"] == "docker_ps" and "empty_reply" in flags


def test_ordinary_tokens_are_not_a_dropped_call():
    c = Completion(text="hello", raw_tokens="hello")
    assert names(detect(req(), c, ttft_s=1, slow_s=30)) == set()


def test_slow_first_token():
    assert "slow" in names(detect(req(), Completion(text="x"), ttft_s=31, slow_s=30))
    assert "slow" not in names(detect(req(), Completion(text="x"), ttft_s=None, slow_s=30))


def test_a_strange_tools_list_does_not_raise():
    assert detect({"tools": "nope"}, Completion(text="x"), ttft_s=1, slow_s=30) == []
    assert names(detect({"tools": [{"function": 3}]}, call("a", "{}"), ttft_s=1, slow_s=30)) == {"unknown_tool"}


@pytest.mark.parametrize("text,expected", [
    ("Let me check that for you.", True), ("I'll look into it now.", True), ("One moment please!", True),
    ("I will run the status check.", True), ("Checking the container now…", True),
    ("The container is up.", False), ("Check the logs yourself with docker logs.", False),
])
def test_promises_action(text, expected):
    assert promises_action(text) is expected
