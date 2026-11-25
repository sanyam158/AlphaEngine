"""Shared FastAPI dependencies: the app context and API-key auth."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from alphaengine.app import AppContext
from alphaengine.portfolio.service import PortfolioService

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def get_ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def get_service(ctx: Annotated[AppContext, Depends(get_ctx)]) -> PortfolioService:
    return PortfolioService(ctx)


def require_auth(
    request: Request,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """With an API key configured, every request must send it as ``X-API-Key``. Without one,
    only requests from this machine are allowed."""
    key = ctx.settings.api_key
    if key is not None:
        if x_api_key is None or not secrets.compare_digest(x_api_key, key.get_secret_value()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")
        return
    host = request.client.host if request.client else ""
    if host not in LOCAL_HOSTS:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "remote access requires ALPHAENGINE_API_KEY to be set",
        )


Ctx = Annotated[AppContext, Depends(get_ctx)]
Service = Annotated[PortfolioService, Depends(get_service)]
