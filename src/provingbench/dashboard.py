"""The dashboard: one page, opened once with the owner's token, which then lives in a cookie."""
import hmac
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from provingbench.auth import COOKIE

STATIC = Path(__file__).parent / "static"
CSP = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
ONE_YEAR_S = 365 * 24 * 3600


async def page(request: Request):
    token = request.query_params.get("token")
    owner = request.app.state.tokens.owner
    if token and hmac.compare_digest(token.encode(), owner.encode()):
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE, owner, max_age=ONE_YEAR_S, httponly=True, samesite="strict")
        return resp
    return FileResponse(STATIC / "index.html", headers={"Content-Security-Policy": CSP, "Cache-Control": "no-store"})


ROUTES = [Route("/", page, methods=["GET"]), Mount("/static", StaticFiles(directory=STATIC), name="static")]
