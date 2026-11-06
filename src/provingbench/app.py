"""The one process: proxy, API and dashboard."""
import contextlib
from datetime import UTC, datetime, timedelta

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route

from provingbench import api, dashboard, proxy
from provingbench.auth import Guard, load_tokens
from provingbench.config import Config
from provingbench.judge import make_judge
from provingbench.runner import RunManager
from provingbench.store import Store
from provingbench.upstreams import Upstreams


def create_app(cfg: Config, *, upstream_transport=None, trigger_transport=None) -> Starlette:
    @contextlib.asynccontextmanager
    async def lifespan(app):
        if cfg.retention_days is not None:
            app.state.store.prune((datetime.now(UTC) - timedelta(days=cfg.retention_days)).isoformat())
        app.state.runs.recover()
        yield
        await app.state.runs.aclose()
        await app.state.ups.aclose()
        app.state.store.close()

    routes = [Route("/a/{agent}/v1/chat/completions", proxy.chat_completions, methods=["POST"]),
              Route("/a/{agent}/v1/models", proxy.list_models, methods=["GET"]), *api.ROUTES, *dashboard.ROUTES]
    tokens = load_tokens(cfg.data_dir)
    app = Starlette(routes=routes, lifespan=lifespan, middleware=[Middleware(Guard, cfg=cfg, tokens=tokens)])
    app.state.tokens = tokens
    app.state.cfg = cfg
    app.state.store = Store(cfg.data_dir / "provingbench.db")
    app.state.ups = Upstreams(cfg, app.state.store, transport=upstream_transport)
    app.state.active = {}
    app.state.runs = RunManager(app.state.store, cfg, app.state.active, trigger_transport=trigger_transport,
                                judge=make_judge(cfg, app.state.ups, app.state.store))
    return app
