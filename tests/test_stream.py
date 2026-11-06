import json

import pytest

from provingbench.stream import Completion, Relay, from_json, render_json, render_sse

TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "system_report", "arguments": '{"a": 1}'}}


def event(data: dict) -> bytes:
    return b"data: " + json.dumps(data).encode() + b"\n\n"


def text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


SSE_TEXT_THEN_TOOL_CALL = b"".join([
    event(text_chunk("H")), event(text_chunk("i")),
    event({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "type": "function", "function": {"name": "system_report", "arguments": '{"a"'}}]}}]}),
    event({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}]}),
    event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    b"data: [DONE]\n\n"])


def split_at_odd_places(data: bytes) -> list[bytes]:
    return [data[i:i + 7] for i in range(0, len(data), 7)]


def test_relay_forwards_bytes_exactly_and_assembles_text_and_calls():
    r = Relay(drop_usage=False)
    out = b"".join(r.feed(c) for c in split_at_odd_places(SSE_TEXT_THEN_TOOL_CALL)) + r.flush()
    assert out == SSE_TEXT_THEN_TOOL_CALL
    c = r.completion()
    assert c.text == "Hi" and c.tool_calls[0]["function"]["name"] == "system_report"
    assert json.loads(c.tool_calls[0]["function"]["arguments"]) == {"a": 1} and c.finish == "tool_calls"


def test_windows_line_endings_are_events_too():
    r = Relay(drop_usage=False)
    data = b'data: {"choices":[{"delta":{"content":"ok"}}]}\r\n\r\n'
    assert r.feed(data) == data and r.completion().text == "ok"


def test_usage_only_event_is_hidden_when_the_agent_did_not_ask():
    r = Relay(drop_usage=True)
    out = r.feed(event({"choices": [], "usage": {"prompt_tokens": 5}}) + b"data: [DONE]\n\n")
    assert b"usage" not in out and b"[DONE]" in out and r.completion().usage == {"prompt_tokens": 5}


def test_usage_on_a_normal_chunk_is_left_alone():
    r = Relay(drop_usage=True)
    data = event({**text_chunk("x"), "usage": {"prompt_tokens": 5}})
    assert r.feed(data) == data and r.completion().usage == {"prompt_tokens": 5}


# Review Focus 3
@pytest.mark.parametrize("garbage", [b"data: {not json\n\n", b"\xff\xfe\n\n", b"event: ping\n\n", b"data: 5\n\n",
                                     b'data: {"choices": 3}\n\n', b'data: {"choices": [{"delta": 7}]}\n\n',
                                     event({"choices": [{"delta": {"content": ["a", "list"]}}]}),
                                     event({"choices": [{"delta": {"tool_calls": "no"}}]}),
                                     event({"choices": [{"delta": {"tool_calls": [{"index": 1}]}}]})])
def test_garbage_is_forwarded_and_never_raises(garbage):
    r = Relay(drop_usage=False)
    assert r.feed(garbage + event(text_chunk("ok"))) == garbage + event(text_chunk("ok"))
    assert r.completion().text == "ok"


def test_stream_without_done_still_gives_a_completion():
    r = Relay(drop_usage=False)
    r.feed(event(text_chunk("par")))
    assert r.completion().text == "par"


def test_an_unfinished_event_is_flushed_at_the_end():
    r = Relay(drop_usage=False)
    assert r.feed(b"data: {tru") == b"" and r.flush() == b"data: {tru"


def test_logprob_tokens_are_collected():
    r = Relay(drop_usage=False)
    r.feed(event({"choices": [{"delta": {}, "logprobs": {"content": [{"token": "<tool_call>"}]}}]}))
    assert r.completion().raw_tokens == "<tool_call>"


def test_first_token_is_noticed_only_for_real_output():
    r = Relay(drop_usage=False)
    r.feed(event({"choices": [{"delta": {"role": "assistant"}}]}))
    assert not r.started
    r.feed(event(text_chunk("x")))
    assert r.started


def test_render_round_trips_through_the_relay():
    original = Completion(text="Hi", tool_calls=[TOOL_CALL], finish="tool_calls")
    r = Relay(drop_usage=False)
    r.feed(render_sse(original, "m"))
    assert r.completion().text == "Hi" and r.completion().tool_calls == [TOOL_CALL]
    assert r.completion().finish == "tool_calls"


def test_render_json_and_from_json_round_trip():
    body = render_json(Completion(text="Hi", tool_calls=[TOOL_CALL], finish="tool_calls", usage={"cost": 1}), "m")
    assert body["choices"][0]["message"]["content"] == "Hi" and body["model"] == "m"
    c = from_json(body)
    assert c.text == "Hi" and c.tool_calls == [TOOL_CALL] and c.finish == "tool_calls" and c.usage == {"cost": 1}


def test_from_json_survives_a_strange_body():
    assert from_json({"choices": "nope"}).text == ""
