# Connecting an agent

Any agent connects in up to two steps. The first gives you model switching and live logging. The second adds test runs.

## 1. Point the agent's model at ProvingBench

Set its OpenAI-compatible base URL to:

```
http://127.0.0.1:7300/a/<agent-id>/v1
```

Pick any `<agent-id>`; it's registered on first use. The agent can send any API key, or none. ProvingBench ignores it and
uses its own keys. Then choose a model with `provingbench use <agent-id> <model>` or on the dashboard's models page.

**If ProvingBench stops, the agent loses its model** unless it has a fallback of its own. Point it back by hand.

**Paid models need your say-so.** A request naming a paid model is refused until that model is on the agent's list, on
the dashboard's Models page. So is one ProvingBench doesn't know the price of. Models you pick yourself don't need listing.

**Model names.** A name with no known prefix (`qwen3:8b`) goes to the default upstream, Ollama unless you set
`[upstreams] default`. A cloud model needs its prefix (`openrouter/x/y`), or a choice made with `provingbench use`.

## 2. For test runs: a test instance and a target

Run a second copy of the agent with its own data folder, its own port and its own agent ID, so test conversations stay
out of its memory and live traffic stays out of the results. Then add a target file in `<provingbench home>/targets/`:

```toml
id = "my-agent-test"
agent_id = "my-agent-test"
url = "http://127.0.0.1:8801/v1/turn"
token_env = "MY_AGENT_TOKEN"            # optional; sent as a bearer token
body = { session_id = "{session}", text = "{message}" }
reply_path = "reply"
timeout_s = 300
```

`{session}` and `{message}` are filled in wherever they appear in `body` or `headers`. `reply_path` is a dotted path to
the reply in the JSON answer, and numbers index into lists (`choices.0.message.content`).

An agent with no HTTP endpoint needs a small adapter that provides one.

## Hermes-Agent

**Status: from its docs, not yet tried.**

Model: in `~/.hermes/.env`, set `OPENAI_BASE_URL` to the ProvingBench address and `OPENAI_API_KEY` to anything. Or in
`config.yaml`, use `provider: custom` with `base_url`.

- Some Hermes tools (vision, web summarization) call OpenRouter on their own, so those calls don't pass through
  ProvingBench and won't be logged.
- One third-party guide says Hermes wants a model with about 64k tokens of context. Smaller local models may fail for
  that reason alone.

Trigger: its API server. Set `API_SERVER_ENABLED=true` and `API_SERVER_KEY`, then run `hermes gateway`.

```toml
id = "hermes-test"
agent_id = "hermes-test"
url = "http://127.0.0.1:8642/v1/chat/completions"
token_env = "HERMES_API_KEY"
headers = { "X-Hermes-Session-Id" = "{session}" }
body = { model = "hermes-agent", messages = [{ role = "user", content = "{message}" }] }
reply_path = "choices.0.message.content"
timeout_s = 300
```

## OpenClaw

**Status: from its docs, not yet tried.**

Model: add a custom provider under `models.providers`, with `baseUrl` set to the ProvingBench address and
`api: "openai-completions"`, and list a model name. The selection in ProvingBench overrides that name.

Trigger: its gateway endpoint, off by default. Set `gateway.http.endpoints.chatCompletions.enabled` to `true`. Its docs
say this endpoint is full operator access, so keep it on loopback.

```toml
id = "openclaw-test"
agent_id = "openclaw-test"
url = "http://127.0.0.1:<gateway port>/v1/chat/completions"
token_env = "OPENCLAW_GATEWAY_TOKEN"
body = { model = "openclaw/default", user = "{session}", messages = [{ role = "user", content = "{message}" }] }
reply_path = "choices.0.message.content"
timeout_s = 300
```
