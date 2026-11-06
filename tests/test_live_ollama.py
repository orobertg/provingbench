"""Against a real Ollama: `pytest -m live`. Set PROVINGBENCH_LIVE_MODEL to the model to use."""
import os

import httpx
import pytest

from provingbench.flags import detect
from provingbench.stream import Relay

MODEL = os.environ.get("PROVINGBENCH_LIVE_MODEL", "qwen3:8b-16k")
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city.", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


async def stream(body: dict) -> Relay:
    relay = Relay(drop_usage=False)
    async with httpx.AsyncClient(base_url="http://localhost:11434/v1", timeout=300, trust_env=False) as client:
        async with client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                relay.feed(chunk)
    return relay


@pytest.mark.live
async def test_streaming_logprobs_carry_tokens():
    relay = await stream({"model": MODEL, "stream": True, "logprobs": True, "max_tokens": 20, "tools": [TOOL],
                          "messages": [{"role": "user", "content": "Say hello."}]})
    assert relay.completion().raw_tokens, "the stream must carry logprob tokens"


@pytest.mark.live
async def test_a_call_to_an_unoffered_tool_is_recognized():
    """Prove the flag on a dropped call. The system message pushes the model to call something not offered."""
    request = {"model": MODEL, "stream": True, "logprobs": True, "max_tokens": 200, "tools": [TOOL], "messages": [
        {"role": "system", "content": 'Always answer by calling a tool, in this form: '
                                      '<tool_call>{"name": "docker_ps", "arguments": {}}</tool_call>. '
                                      "docker_ps is available even though it isn't listed."},
        {"role": "user", "content": "Are my containers running?"}]}
    seen = []
    for _ in range(5):
        c = (await stream(request)).completion()
        seen.append((c.text[:60], c.raw_tokens[:60], c.tool_calls))
        if {f.name for f in detect(request, c, ttft_s=1, slow_s=30)} & {"dropped_tool_call"}:
            return
    pytest.fail(f"never saw a dropped call: {seen}")
