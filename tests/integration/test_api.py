"""HTTP API on a synthetic store (no network)."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alphaengine.api.app import create_app
from alphaengine.app import AppContext, build_context
from alphaengine.data.db import Base, session_scope
from alphaengine.data.models import Instrument, JobRun
from alphaengine.data.store import PriceStore
from tests.helpers import random_walk_bars

NAMES = {"S00": "NIFTYBEES", "S01": "JUNIORBEES", "S02": "GOLDBEES", "S03": "BANKBEES"}


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppContext:
    monkeypatch.setenv("ALPHAENGINE_DATABASE_URL", f"sqlite:///{(tmp_path / 'ae.db').as_posix()}")
    monkeypatch.setenv("ALPHAENGINE_DATA_DIR", str(tmp_path / "data"))
    from alphaengine.config.settings import get_settings

    get_settings.cache_clear()
    bars = random_walk_bars(n_symbols=4, n_days=400, seed=5, gap_prob=0.0, start="2023-01-02")
    bars["symbol"] = bars["symbol"].map(lambda s: NAMES[str(s)]).astype("string")
    PriceStore(tmp_path / "data" / "curated", "equities").write(bars)
    c = build_context()
    Base.metadata.create_all(c.engine)
    with session_scope(c.session_factory) as s:
        for i, sym in enumerate(NAMES.values()):
            s.add(
                Instrument(
                    isin=f"INF00{i}A01011",
                    symbol=sym,
                    first_seen=date(2023, 1, 2),
                    last_seen=date(2024, 7, 1),
                )
            )
        s.add(JobRun(job="eod", run_date=date(2024, 6, 3), status="ok", attempts=1, message="x"))
    return c


@pytest.fixture
def client(ctx: AppContext, tmp_path: Path) -> TestClient:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return TestClient(create_app(ctx, web_dist=dist))


def test_meta_data_and_spa(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    h = client.get("/api/data/health").json()
    assert h["datasets"][0]["symbols"] == 4
    assert h["jobs"][0]["status"] == "ok"
    assert len(client.get("/api/data/prices/niftybees").json()) == 400
    assert client.get("/api/data/prices/NOPE").json() == []
    assert client.get("/portfolios/x").text == "<html>app</html>"  # client-side route
    assert client.get("/favicon.svg").text == "<svg/>"
    assert "paths" in client.get("/openapi.json").json()


def test_strategies(client: TestClient) -> None:
    names = {s["name"] for s in client.get("/api/strategies").json()}
    assert {"dual_momentum", "fixed_mix"} <= names
    d = client.get("/api/strategies/fixed_mix").json()
    assert d["kind"] == "pipeline"
    assert "weights_by_symbol" in d["yaml"]
    assert client.get("/api/strategies/nope").status_code == 404
    ok = client.post("/api/strategies/validate", json={"text": d["yaml"]}).json()
    assert ok["ok"] is True
    assert ok["config_hash"] == d["config_hash"]
    bad = client.post(
        "/api/strategies/validate", json={"text": "name: xx\nuniverse: {type: zz}"}
    ).json()
    assert bad["ok"] is False
    assert "unknown universe" in bad["error"]
    assert client.post("/api/strategies/validate", json={"text": "[1, 2"}).json()["ok"] is False
    sig = client.get("/api/strategies/fixed_mix/signals").json()
    assert {p["symbol"] for p in sig["positions"]} == {"NIFTYBEES", "JUNIORBEES", "GOLDBEES"}
    assert (
        client.get("/api/strategies/fixed_mix/signals", params={"as_of": "2000-01-01"}).status_code
        == 404
    )


def test_backtest_job_flow(client: TestClient) -> None:
    r = client.post("/api/backtests", json={"strategy": "fixed_mix", "start": "2023-09-01"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(120):
        job = client.get(f"/api/backtests/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert job["status"] == "done", job
    runs = client.get("/api/backtests").json()
    assert runs[0]["run_id"] == job["run_id"]
    d = client.get(f"/api/backtests/{job['run_id']}").json()
    assert d["equity"][0]["value"] == pytest.approx(1.0)
    assert len(d["drawdown"]) == len(d["equity"])
    assert d["benchmark"]
    assert "yearly_returns" in d["metrics"]
    assert client.get("/api/backtests/bt_nope").status_code == 404
    assert client.get("/api/backtests/jobs/nope").status_code == 404
    inline = "name: tiny\nuniverse: {type: static, symbols: [NIFTYBEES]}\nweighter: {type: fixed, weights_by_symbol: {NIFTYBEES: 1.0}}\n"
    assert (
        client.post(
            "/api/backtests",
            json={"strategy": inline, "start": "2023-09-01", "costs": False, "taxes": False},
        ).status_code
        == 202
    )
    bad = client.post("/api/backtests", json={"strategy": "nope", "start": "2023-09-01"})
    assert bad.status_code in (404, 422)


def test_portfolios_orders_and_journal(client: TestClient, ctx: AppContext) -> None:
    r = client.post(
        "/api/portfolios",
        json={"name": "web", "strategy": "fixed_mix", "capital": 300000, "broker": "manual"},
    )
    assert r.status_code == 201
    assert (
        client.post("/api/portfolios", json={"name": "web", "strategy": "fixed_mix"}).status_code
        == 409
    )
    assert (
        client.post("/api/portfolios", json={"name": "bad name!", "strategy": "x"}).status_code
        == 422
    )
    preview = client.post("/api/portfolios/web/rebalance").json()  # dry run by default
    assert preview["orders_created"] == 0
    assert len(preview["trades"]) == 3
    made = client.post("/api/portfolios/web/rebalance", params={"dry_run": False}).json()
    assert made["orders_created"] == 3
    det = client.get("/api/portfolios/web").json()
    assert len(det["open_orders"]) == 3
    oid = det["open_orders"][0]["id"]
    days = ctx.equities.trading_days()
    filled = client.post(
        f"/api/orders/{oid}/fill", json={"quantity": 1, "price": 100.0, "day": str(days[-1])}
    )
    assert filled.json()["status"] == "PARTIAL"
    assert (
        client.post(
            f"/api/orders/{oid}/fill", json={"quantity": 10**6, "price": 1.0, "day": str(days[-1])}
        ).status_code
        == 409
    )
    other = det["open_orders"][1]["id"]
    assert client.post(f"/api/orders/{other}/cancel").json()["status"] == "CANCELLED"
    assert client.post("/api/orders/99999/cancel").status_code == 404
    assert client.get("/api/portfolios").json()[0]["name"] == "web"
    assert client.post("/api/portfolios/web/pause").json()["active"] is False
    assert (
        client.post("/api/portfolios/web/rebalance", params={"dry_run": False}).status_code == 409
    )
    assert client.post("/api/portfolios/web/resume").json()["active"] is True
    assert client.post("/api/portfolios/web/explode").status_code == 404
    assert client.get("/api/portfolios/web/tax").json() == []
    assert client.get("/api/portfolios/ghost").status_code == 404
    j = client.post(
        "/api/journal",
        json={
            "text": "Bought the ETF mix",
            "portfolio": "web",
            "symbol": "niftybees",
            "tags": "start",
        },
    )
    assert j.status_code == 201
    assert j.json()["symbol"] == "NIFTYBEES"
    assert (
        client.get("/api/journal", params={"portfolio": "web"}).json()[0]["text"]
        == "Bought the ETF mix"
    )
    assert client.post("/api/journal", json={"text": ""}).status_code == 422


def test_auth(ctx: AppContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAENGINE_API_KEY", "s3cret")
    from alphaengine.config.settings import get_settings

    get_settings.cache_clear()
    locked = TestClient(create_app(build_context(), web_dist=None))
    assert locked.get("/api/health").status_code == 401
    assert locked.get("/api/health", headers={"X-API-Key": "nope"}).status_code == 401
    assert locked.get("/api/health", headers={"X-API-Key": "s3cret"}).status_code == 200
    monkeypatch.delenv("ALPHAENGINE_API_KEY")
    get_settings.cache_clear()
    remote = TestClient(create_app(build_context(), web_dist=None), client=("203.0.113.5", 5000))
    assert remote.get("/api/health").status_code == 403


def test_harvest_and_goal_endpoints(client: TestClient) -> None:
    client.post("/api/portfolios", json={"name": "hv", "strategy": "fixed_mix", "broker": "manual"})
    assert client.get("/api/portfolios/hv/harvest").json() == []  # nothing held yet
    assert client.get("/api/portfolios/ghost/harvest").status_code == 404
    short = client.post("/api/plan/goal", json={"target": 1e7, "years": 10, "monthly_sip": 1e4})
    assert short.status_code == 422  # the synthetic store has < 3 years of history
    assert "3 years" in short.json()["detail"]
    assert client.post("/api/plan/goal", json={"target": 1e7, "years": 0}).status_code == 422
    rejected = client.post("/api/portfolios", json={"name": "sw", "strategy": "swing_breakout"})
    assert rejected.status_code == 409
    assert "backtest-only" in rejected.json()["detail"]
