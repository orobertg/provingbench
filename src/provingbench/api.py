"""The JSON API, shared by the dashboard, the command line and agents."""
import json

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from provingbench import __version__
from provingbench.profile import profile, scoreboard
from provingbench.proxy import valid_agent
from provingbench.runner import RunRefused, RunSpec
from provingbench.suite import SuiteError, load_suite
from provingbench.target import TargetError, load_targets
from provingbench.upstreams import is_local, is_paid, route


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def endpoint(fn):
    """Turn an ApiError into its JSON answer, and a returned dict into a 200."""
    async def wrapped(request: Request):
        try:
            result = await fn(request)
        except ApiError as e:
            return JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=e.status)
        return result if not isinstance(result, dict) else JSONResponse(result)
    return wrapped


async def json_object(request: Request) -> dict:
    try:
        body = json.loads(await request.body())
    except (ValueError, RecursionError):
        body = None
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", "the body must be a JSON object")
    return body


def owner_only(request: Request) -> None:
    if request.state.role != "owner":
        raise ApiError(403, "forbidden", "only the owner can do this")


def _agent_id(request: Request) -> str:
    agent = request.path_params["id"]
    if not valid_agent(agent):
        raise ApiError(404, "not_found", "no such agent ID")
    return agent


def check_may_use(request: Request, agent: str, model: str) -> None:
    """An agent's token reaches local and free models, and the paid ones the owner put on its list."""
    app = request.app
    if request.state.role != "owner" and is_paid(app.state.cfg, app.state.store, model) \
            and model not in app.state.store.allowed_models(agent):
        raise ApiError(403, "forbidden", f"{model} is a paid model that isn't on this agent's list")


@endpoint
async def models(request: Request):
    cfg = request.app.state.cfg
    return {"models": [{**m, "cloud": not is_local(route(cfg, m["id"])[0])} for m in await request.app.state.ups.listing()]}


@endpoint
async def agents(request: Request):
    return {"agents": request.app.state.store.agents()}


@endpoint
async def get_model(request: Request):
    app = request.app
    model = app.state.store.selection(_agent_id(request))
    return {"model": model, "cloud": bool(model) and not is_local(route(app.state.cfg, model)[0])}


@endpoint
async def put_model(request: Request):
    agent = _agent_id(request)
    body = await json_object(request)
    if "model" not in body or not (body["model"] is None or isinstance(body["model"], str)):
        raise ApiError(400, "bad_request", "model must be a model ID, or null to clear the choice")
    model = body["model"]
    if agent in request.app.state.active:
        raise ApiError(409, "conflict", f"{agent} is in a test run; wait for it to end or stop it")
    if model:
        check_may_use(request, agent, model)
    request.app.state.store.select(agent, model, request.state.role)
    return {"model": model, "cloud": bool(model) and not is_local(route(request.app.state.cfg, model)[0])}


@endpoint
async def put_allowed(request: Request):
    owner_only(request)
    agent = _agent_id(request)
    body = await json_object(request)
    names = body.get("models")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ApiError(400, "bad_request", "models must be a list of model IDs")
    request.app.state.store.set_allowed(agent, names)
    return {"models": names}


def number(request: Request, name: str, default: int) -> int:
    try:
        return int(request.query_params.get(name, default))
    except ValueError:
        raise ApiError(400, "bad_request", f"{name} must be a number") from None


@endpoint
async def calls(request: Request):
    q = request.query_params
    rows = request.app.state.store.calls(after=number(request, "after", 0), agent=q.get("agent"), model=q.get("model"),
                                         flag=q.get("flag"), limit=min(number(request, "limit", 100), 500))
    return {"calls": rows}


@endpoint
async def call(request: Request):
    try:
        row = request.app.state.store.call(int(request.path_params["id"]))
    except ValueError:
        row = None
    if row is None:
        raise ApiError(404, "not_found", "no such call")
    return row


@endpoint
async def status(request: Request):
    return {"log_errors": request.app.state.store.log_errors, "version": __version__}


def _run_spec(request: Request, body: dict) -> RunSpec:
    cfg = request.app.state.cfg
    names, repeats = body.get("models"), body.get("repeats")
    limit = body.get("max_cost_usd", cfg.max_cost_usd)
    if not isinstance(body.get("target"), str) or not isinstance(body.get("suite"), str):
        raise ApiError(400, "bad_request", "target and suite must be names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ApiError(400, "bad_request", "models must be a list of model IDs")
    if repeats is not None and (not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1):
        raise ApiError(400, "bad_request", "repeats must be a whole number of at least 1")
    if not isinstance(limit, int | float) or isinstance(limit, bool) or limit < 0:
        raise ApiError(400, "bad_request", "max_cost_usd must be a number, zero or more")
    return RunSpec(body["target"], body["suite"], names, repeats, limit, str(body.get("label", "")))


def _refused(e: RunRefused) -> ApiError:
    return ApiError(409 if e.conflict else 400, "conflict" if e.conflict else "bad_request", str(e))


def _check_agent_may_run(request: Request, spec: RunSpec) -> None:
    if request.state.role == "owner":
        return
    if spec.max_cost_usd > request.app.state.cfg.max_cost_usd:
        raise ApiError(403, "forbidden", "only the owner can raise a run's cost limit")
    try:
        agent_id = load_targets(request.app.state.cfg.data_dir / "targets")[spec.target].agent_id
    except (KeyError, TargetError):
        return      # starting it will refuse with the reason
    for model in spec.models:
        check_may_use(request, agent_id, model)
    _check_agent_may_use_judge(request, spec, agent_id)


def _check_agent_may_use_judge(request: Request, spec: RunSpec, agent_id: str) -> None:
    """A judged task spends on the judge's model too."""
    cfg, store = request.app.state.cfg, request.app.state.store
    if not cfg.judge_model or not is_paid(cfg, store, cfg.judge_model) or cfg.judge_model in store.allowed_models(agent_id):
        return
    try:
        judged = any(c["type"] == "judge" for t in load_suite(cfg.suites[spec.suite]) for c in t.checks)
    except (KeyError, SuiteError):
        return      # starting it will refuse with the reason
    if judged:
        raise ApiError(403, "forbidden", f"this suite uses the judge, {cfg.judge_model}, a paid model not on this agent's list")


@endpoint
async def targets(request: Request):
    try:
        found = load_targets(request.app.state.cfg.data_dir / "targets")
    except TargetError as e:
        raise ApiError(500, "bad_target", str(e)) from e
    return {"targets": [{"id": t.id, "agent_id": t.agent_id, "url": t.url} for t in found.values()]}


def _suite_row(name: str, path) -> dict:
    try:
        return {"name": name, "tasks": len(load_suite(path)), "error": None}
    except SuiteError as e:
        return {"name": name, "tasks": 0, "error": str(e)}


@endpoint
async def suites(request: Request):
    return {"suites": [_suite_row(name, path) for name, path in request.app.state.cfg.suites.items()]}


@endpoint
async def estimate(request: Request):
    spec = _run_spec(request, await json_object(request))
    try:
        return {"estimate_usd": request.app.state.runs.estimate(spec)}
    except RunRefused as e:
        raise _refused(e) from e


@endpoint
async def start_run(request: Request):
    spec = _run_spec(request, await json_object(request))
    _check_agent_may_run(request, spec)
    try:
        return JSONResponse({"id": request.app.state.runs.start(spec)}, status_code=202)
    except RunRefused as e:
        raise _refused(e) from e


@endpoint
async def list_runs(request: Request):
    app = request.app
    rows = app.state.store.runs(limit=min(number(request, "limit", 50), 200))
    return {"runs": [{**r, "state": app.state.runs.live_state(r)} for r in rows]}


def _row_id(request: Request) -> int:
    try:
        return int(request.path_params["id"])
    except ValueError:
        raise ApiError(404, "not_found", "no such row") from None


@endpoint
async def get_run(request: Request):
    store, run_id = request.app.state.store, _row_id(request)
    row = store.run(run_id)
    if row is None:
        raise ApiError(404, "not_found", "no such run")
    overrides = store.overrides()
    results = [{**r, "override": overrides.get(r["id"])} for r in store.results(run_id)]
    return {**row, "state": request.app.state.runs.live_state(row), "spent_usd": store.spend(run_id), "results": results}


@endpoint
async def get_result(request: Request):
    store = request.app.state.store
    row = store.result(_row_id(request))
    if row is None:
        raise ApiError(404, "not_found", "no such result")
    return {**row, "override": store.overrides().get(row["id"]), "calls": store.attempt_calls(row["attempt_id"])}


@endpoint
async def override_result(request: Request):
    owner_only(request)
    store = request.app.state.store
    result_id, body = _row_id(request), await json_object(request)
    if store.result(result_id) is None:
        raise ApiError(404, "not_found", "no such result")
    if body.get("outcome") not in ("pass", "fail") or not (isinstance(body.get("note"), str) and body["note"].strip()):
        raise ApiError(400, "bad_request", "an override needs an outcome of pass or fail, and a note saying why")
    store.add_override(result_id, body["outcome"], body["note"].strip())
    return {"outcome": body["outcome"]}


@endpoint
async def put_notes(request: Request):
    owner_only(request)
    body = await json_object(request)
    if not isinstance(body.get("text"), str):
        raise ApiError(400, "bad_request", "text must be some text")
    request.app.state.store.note(request.path_params["id"], body["text"])
    return {"text": body["text"]}


@endpoint
async def stop_run(request: Request):
    try:
        request.app.state.runs.stop(_row_id(request))
    except RunRefused as e:
        raise _refused(e) from e
    return {"stopping": True}


@endpoint
async def resume_run(request: Request):
    owner_only(request)
    body = await json_object(request)
    limit = body.get("max_cost_usd")
    if limit is not None and (not isinstance(limit, int | float) or isinstance(limit, bool) or limit < 0):
        raise ApiError(400, "bad_request", "max_cost_usd must be a number, zero or more")
    try:
        request.app.state.runs.resume(_row_id(request), limit)
    except RunRefused as e:
        raise _refused(e) from e
    return {"resumed": True}


@endpoint
async def board(request: Request):
    return scoreboard(request.app.state.store)


@endpoint
async def model_profile(request: Request):
    return profile(request.app.state.store, request.path_params["id"])


ROUTES = [Route("/api/v1/scoreboard", board, methods=["GET"]),
          Route("/api/v1/models/{id:path}/profile", model_profile, methods=["GET"]),
          Route("/api/v1/targets", targets, methods=["GET"]),
          Route("/api/v1/suites", suites, methods=["GET"]),
          Route("/api/v1/estimate", estimate, methods=["POST"]),
          Route("/api/v1/runs", list_runs, methods=["GET"]),
          Route("/api/v1/runs", start_run, methods=["POST"]),
          Route("/api/v1/runs/{id}", get_run, methods=["GET"]),
          Route("/api/v1/runs/{id}/stop", stop_run, methods=["POST"]),
          Route("/api/v1/runs/{id}/resume", resume_run, methods=["POST"]),
          Route("/api/v1/results/{id}", get_result, methods=["GET"]),
          Route("/api/v1/results/{id}/override", override_result, methods=["POST"]),
          Route("/api/v1/models/{id:path}/notes", put_notes, methods=["PUT"]),
          Route("/api/v1/models", models, methods=["GET"]),
          Route("/api/v1/agents", agents, methods=["GET"]),
          Route("/api/v1/agents/{id}/model", get_model, methods=["GET"]),
          Route("/api/v1/agents/{id}/model", put_model, methods=["PUT"]),
          Route("/api/v1/agents/{id}/allowed-models", put_allowed, methods=["PUT"]),
          Route("/api/v1/calls", calls, methods=["GET"]),
          Route("/api/v1/calls/{id}", call, methods=["GET"]),
          Route("/api/v1/status", status, methods=["GET"])]
