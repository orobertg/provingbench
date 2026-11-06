"""`provingbench`: setup, serve, doctor, and a thin front to the running server's API."""
import argparse
import os
import socket
import sys
import time
from dataclasses import replace

import httpx

from provingbench.auth import load_tokens
from provingbench.config import Config, ConfigError, home, load_config
from provingbench.upstreams import is_local

NOT_RUNNING = "ProvingBench isn't running; start it with: provingbench serve"


def err(text: str) -> None:
    print(text, file=sys.stderr)


def _reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url}/models", timeout=3, trust_env=False).status_code < 500
    except httpx.HTTPError:
        return False


def _upstream_lines(cfg: Config) -> tuple[list[str], bool]:
    lines, healthy = [], True
    for up in cfg.upstreams.values():
        if is_local(up):
            ok = _reachable(up.base_url)
            healthy = healthy and ok
            lines.append(f"{up.name}: {'ok' if ok else 'not reachable'} at {up.base_url}")
        elif up.api_key_env and os.environ.get(up.api_key_env):
            lines.append(f"{up.name}: key set ({up.api_key_env})")
        else:
            lines.append(f"{up.name}: no key (set {up.api_key_env} to use it)")
    return lines, healthy


def setup() -> int:
    path = home() / "config.toml"
    if path.exists():
        print(f"{path} already exists; leaving it as it is")
    else:
        name = input("Environment variable holding your OpenRouter key [OPENROUTER_API_KEY]: ").strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[server]\nhost = \"127.0.0.1\"\nport = 7300\n\n[upstreams.openrouter]\n"
                        "base_url = \"https://openrouter.ai/api/v1\"\n"
                        f"api_key_env = \"{name or 'OPENROUTER_API_KEY'}\"\n", encoding="utf-8")
        print(f"wrote {path}")
    cfg = load_config()
    print("\n".join(_upstream_lines(cfg)[0]))
    print("next: provingbench serve")
    return 0


def doctor() -> int:
    cfg = load_config()
    lines, healthy = _upstream_lines(cfg)
    print("\n".join(lines))
    print(f"data folder: {cfg.data_dir}")
    return 0 if healthy else 1


def _sign_in(cfg: Config) -> str:
    return f"http://{cfg.host}:{cfg.port}/?token={load_tokens(cfg.data_dir).owner}"


def serve(cfg: Config, port: int | None) -> int:
    import uvicorn

    from provingbench.app import create_app
    if port:
        cfg = replace(cfg, port=port)
    with socket.socket() as probe:
        try:
            probe.bind((cfg.host, cfg.port))
        except OSError:
            err(f"port {cfg.port} is already in use; is ProvingBench already running? (try --port)")
            return 1
    print(f"ProvingBench on http://{cfg.host}:{cfg.port}")
    print(f"dashboard: {_sign_in(cfg)}")
    print(f"point an agent's base URL at: http://{cfg.host}:{cfg.port}/a/<agent-id>/v1", flush=True)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning")
    return 0


class _Refused(Exception):
    pass


def call_api(cfg: Config, method: str, path: str, body: dict | None = None) -> dict:
    token = load_tokens(cfg.data_dir).owner
    try:
        resp = httpx.request(method, f"http://{cfg.host}:{cfg.port}/api/v1{path}", timeout=60, trust_env=False,
                             headers={"Authorization": f"Bearer {token}"}, **({"json": body} if body is not None else {}))
    except httpx.TransportError:
        raise _Refused(NOT_RUNNING) from None
    try:
        data = resp.json()
    except ValueError:
        raise _Refused(f"HTTP {resp.status_code}: {resp.text[:200]}") from None
    if resp.status_code >= 400:
        raise _Refused(data.get("error", {}).get("message", f"HTTP {resp.status_code}"))
    return data


def _price(model: dict) -> str:
    prompt, completion = model.get("price_prompt"), model.get("price_completion")
    if prompt is None or completion is None:
        return "price unknown"
    if not prompt and not completion:
        return "free"
    return f"${prompt * 1e6:.2f}/${completion * 1e6:.2f} per million tokens"


def models(cfg: Config) -> int:
    for m in call_api(cfg, "GET", "/models")["models"]:
        tools = {True: "tools", False: "no tools", None: ""}[m.get("supports_tools")]
        print(f"{m['id']}  {_price(m)}  {tools}".rstrip())
    return 0


def use(cfg: Config, agent: str, model: str) -> int:
    call_api(cfg, "PUT", f"/agents/{agent}/model", {"model": model})
    print(f"{agent} now uses {model}")
    return 0


def run(cfg: Config, args) -> int:
    body = {"target": args.target, "suite": args.suite, "models": args.model}
    for key, value in (("repeats", args.repeats), ("max_cost_usd", args.max_cost), ("label", args.label)):
        if value is not None:
            body[key] = value
    estimate = call_api(cfg, "POST", "/estimate", body)["estimate_usd"]
    print("Estimated cost: unknown, because a model's price isn't known" if estimate is None
          else f"Estimated cost: about ${estimate:.2f}")
    run_id = call_api(cfg, "POST", "/runs", body)["id"]
    print(f"run {run_id} started; follow it on the dashboard or press Ctrl-C to leave it going")
    seen = -1
    while True:
        state = call_api(cfg, "GET", f"/runs/{run_id}")
        if len(state["results"]) != seen:
            seen = len(state["results"])
            print(f"{seen} of {state['total']} attempts done", flush=True)
        if state["state"] != "running":
            break
        time.sleep(2)
    by_model: dict[str, list[str]] = {}
    for result in state["results"]:
        by_model.setdefault(result["model"], []).append(result["outcome"])
    for model, outcomes in by_model.items():
        print(f"{model}: {outcomes.count('pass')} of {len(outcomes) - outcomes.count('error')} passed"
              + (f", {outcomes.count('error')} couldn't be graded" if "error" in outcomes else ""))
    print(f"{state['state']}, spent ${state['spent_usd']:.2f}")
    if state.get("error"):
        err(f"error: {state['error']}")
    return 0 if state["state"] == "done" else 1


def results(cfg: Config) -> int:
    board = call_api(cfg, "GET", "/scoreboard")
    if not board["cells"]:
        print("No results yet. Start a run with: provingbench run")
        return 0
    cells = {(c["model"], c["task_id"]): c for c in board["cells"]}
    for model in board["models"]:
        print(model)
        for task in board["tasks"]:
            cell = cells.get((model, task["id"]))
            if cell:
                note = " (task changed since)" if cell["stale"] else ""
                print(f"  {task['id']}: {cell['passed']}/{cell['passed'] + cell['failed']}{note}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="provingbench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="write a config and check the upstreams")
    sub.add_parser("doctor", help="check the upstreams")
    sub.add_parser("dashboard", help="print the dashboard's sign-in address")
    sub.add_parser("models", help="list available models")
    serve_p = sub.add_parser("serve", help="start ProvingBench")
    serve_p.add_argument("--port", type=int)
    use_p = sub.add_parser("use", help="switch an agent's model")
    use_p.add_argument("agent")
    use_p.add_argument("model")
    run_p = sub.add_parser("run", help="run a suite against one or more models")
    run_p.add_argument("--target", required=True)
    run_p.add_argument("--suite", required=True)
    run_p.add_argument("--model", required=True, action="append")
    run_p.add_argument("--repeats", type=int)
    run_p.add_argument("--max-cost", type=float)
    run_p.add_argument("--label")
    sub.add_parser("results", help="show the scoreboard")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "setup":
            return setup()
        if args.command == "doctor":
            return doctor()
        cfg = load_config()
        if args.command == "serve":
            return serve(cfg, args.port)
        if args.command == "dashboard":
            print(_sign_in(cfg))
            return 0
        if args.command == "models":
            return models(cfg)
        if args.command == "run":
            return run(cfg, args)
        if args.command == "results":
            return results(cfg)
        return use(cfg, args.agent, args.model)
    except ConfigError as e:
        err(f"error[config]: {e}")
    except _Refused as e:
        err(f"error: {e}")
    return 1
