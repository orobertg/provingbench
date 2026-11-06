import json
import tomllib
from pathlib import Path

import httpx
import pytest

from provingbench.target import Target, TargetError, load_targets, send


class Seen:
    def __init__(self, response=None):
        self.bodies, self.requests = [], []
        self.response = response

    def __call__(self, request):
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response or httpx.Response(200, json={"reply": "echoed"})

    @property
    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def echo_target(token_env="", response=None, **over):
    fields = {"id": "t", "agent_id": "bot", "url": "http://agent.test/turn", "token_env": token_env,
              "body": {"session_id": "{session}", "text": "{message}"}, "headers": {}, "reply_path": "reply",
              "timeout_s": 30, **over}
    return Target(**fields), Seen(response)


async def test_send_fills_the_template_without_breaking_on_quotes():
    target, seen = echo_target()
    reply = await send(target, seen.client, "s1", 'say "hi"\n')
    assert seen.bodies[-1] == {"session_id": "s1", "text": 'say "hi"\n'} and reply.reply == "echoed" and reply.error is None


async def test_a_message_containing_a_placeholder_is_left_alone():
    target, seen = echo_target()
    await send(target, seen.client, "s1", "what does {session} mean?")
    assert seen.bodies[-1]["text"] == "what does {session} mean?"


async def test_send_sends_the_token_from_the_named_variable(monkeypatch):
    monkeypatch.setenv("TOK", "abc")
    target, seen = echo_target(token_env="TOK")
    await send(target, seen.client, "s", "m")
    assert seen.requests[-1].headers["authorization"] == "Bearer abc"


async def test_chat_style_agents_work_with_nested_bodies_headers_and_list_paths():
    target, seen = echo_target(
        body={"model": "agent", "messages": [{"role": "user", "content": "{message}"}]},
        headers={"X-Session": "{session}"}, reply_path="choices.0.message.content",
        response=httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]}))
    reply = await send(target, seen.client, "s9", "hi")
    assert seen.bodies[-1]["messages"][0]["content"] == "hi" and seen.requests[-1].headers["x-session"] == "s9"
    assert reply.reply == "hello"


@pytest.mark.parametrize("response,fragment", [
    (httpx.Response(502, json={"error": {"message": "boom"}}), "502"),
    (httpx.Response(200, json={"other": 1}), "reply"),
    (httpx.Response(200, text="not json"), "JSON"),
    (httpx.Response(200, json={"reply": 5}), "reply"),
    (httpx.ConnectError("refused"), "reach"),
])
async def test_send_turns_failures_into_an_error(response, fragment):
    target, seen = echo_target(response=response)
    reply = await send(target, seen.client, "s", "m")
    assert fragment in reply.error and reply.reply is None


async def test_a_connection_failure_is_marked_as_unreachable():
    target, seen = echo_target(response=httpx.ConnectError("refused"))
    assert (await send(target, seen.client, "s", "m")).unreachable


def good(tmp_path, name="a", text=None):
    (tmp_path / f"{name}.toml").write_text(text or f'id = "{name}"\nagent_id = "bot"\nurl = "http://x/t"\n'
                                           'reply_path = "reply"\nbody = {{ text = "{{message}}" }}\n'.replace("{{", "{").replace("}}", "}"),
                                           encoding="utf-8")


def test_load_targets_by_id_with_defaults(tmp_path):
    good(tmp_path)
    target = load_targets(tmp_path)["a"]
    assert target.agent_id == "bot" and target.timeout_s == 300 and target.token_env == "" and target.headers == {}


def test_no_folder_means_no_targets(tmp_path):
    assert load_targets(tmp_path / "nope") == {}


@pytest.mark.parametrize("text,problem", [
    ("id = ", "bad.toml"),
    ('id = "x"\n', "agent_id"),
    ('id = "x"\nagent_id = "bad id"\nurl = "http://x"\nreply_path = "r"\nbody = {a = "b"}\n', "agent_id"),
    ('id = "x"\nagent_id = "b"\nurl = "ftp://x"\nreply_path = "r"\nbody = {a = "b"}\n', "url"),
    ('id = "x"\nagent_id = "b"\nurl = "http://x"\nreply_path = "r"\nbody = {a = "no placeholder"}\n', "message"),
])
def test_a_bad_target_names_itself(tmp_path, text, problem):
    good(tmp_path, "bad", text)
    with pytest.raises(TargetError, match=problem) as e:
        load_targets(tmp_path)
    assert "bad.toml" in str(e.value)


def test_the_shipped_example_is_valid(tmp_path):
    source = Path(__file__).parent.parent / "examples" / "target.toml"
    assert tomllib.loads(source.read_text(encoding="utf-8"))["id"]
    (tmp_path / "target.toml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_targets(tmp_path)
