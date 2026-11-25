"""FastAPI application: JSON API under ``/api`` and the built dashboard (``web/dist``) at ``/``.

Run with ``alphaengine serve`` (binds to 127.0.0.1 by default).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from alphaengine import __version__
from alphaengine.api.deps import require_auth
from alphaengine.api.routers import backtests, data, portfolios, strategies
from alphaengine.api.routers.backtests import JobBoard
from alphaengine.api.schemas import Health
from alphaengine.app import AppContext, build_context

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def create_app(ctx: AppContext | None = None, *, web_dist: Path | None = WEB_DIST) -> FastAPI:
    app = FastAPI(
        title="AlphaEngine",
        version=__version__,
        description="Research, backtesting and paper trading for NSE.",
    )
    app.state.ctx = ctx or build_context()
    app.state.backtests = JobBoard()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # vite dev server
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

    @api.get("/health", response_model=Health, tags=["meta"])
    def health() -> Health:
        return Health(version=__version__)

    for module in (data, strategies, backtests, portfolios):
        api.include_router(module.router)
    app.include_router(api)

    if web_dist is not None and (web_dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:  # client-side routes all serve index.html
            target = web_dist / path
            if path and target.is_file() and web_dist in target.resolve().parents:
                return FileResponse(target)
            return FileResponse(web_dist / "index.html")

    return app
