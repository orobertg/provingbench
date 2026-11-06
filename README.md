# ProvingBench

Find out which model actually works for your agent.

ProvingBench sits between an agent and its model. It switches models on the fly, logs every call per model,
runs task lists you write yourself, grades the results, and shows them on a dashboard. Any agent that can
set an OpenAI-compatible base URL can connect.

```
pip install -e .
provingbench setup
provingbench serve
```

Then point your agent's base URL at `http://127.0.0.1:7300/a/<agent-id>/v1`.
See `docs/connecting-agents.md`.
