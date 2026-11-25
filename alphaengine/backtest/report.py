"""Self-contained HTML (inline SVG, no scripts) and JSON reports for a backtest run."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaengine.backtest.metrics import drawdown, growth_index, to_jsonable, twr_returns


def _pct(x: Any, digits: int = 1) -> str:
    return (
        "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}%}"
    )


def _num(x: Any, digits: int = 2) -> str:
    return (
        "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{digits}f}"
    )


def _inr(x: Any) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    v = float(x)
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.0f}"


def _line_chart(
    series: dict[str, pd.Series], *, log: bool = False, height: int = 260, pct_axis: bool = False
) -> str:
    """Minimal responsive SVG line chart. ``series`` values share one y-axis."""
    w, h, pad_l, pad_b, pad_t = 900, height, 64, 28, 12
    frames = [s.dropna() for s in series.values() if not s.dropna().empty]
    if not frames:
        return ""
    x0 = min(f.index[0] for f in frames)
    x1 = max(f.index[-1] for f in frames)
    span = max((x1 - x0).days, 1)
    vals = np.concatenate([f.to_numpy(dtype=float) for f in frames])
    tv = np.log(vals) if log else vals
    lo, hi = float(np.nanmin(tv)), float(np.nanmax(tv))
    if hi - lo < 1e-12:
        hi = lo + 1

    def sx(d: pd.Timestamp) -> float:
        return float(pad_l + (d - x0).days / span * (w - pad_l - 8))

    def sy(v: float) -> float:
        tv = math.log(v) if log else v
        return float(pad_t + (hi - tv) / (hi - lo) * (h - pad_t - pad_b))

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for k in range(5):
        level = lo + (hi - lo) * k / 4
        y = sy(math.exp(level) if log else level)
        val = math.exp(level) if log else level
        label = f"{val:.0%}" if pct_axis else (f"{val:,.2f}" if val < 100 else f"{val:,.0f}")
        parts.append(f'<line x1="{pad_l}" x2="{w - 8}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">{label}</text>'
        )
    for year in range(x0.year + 1, x1.year + 1):
        d = pd.Timestamp(year=year, month=1, day=1)
        if (x1.year - x0.year) > 12 and year % 2:
            continue
        x = sx(d)
        parts.append(
            f'<text x="{x:.1f}" y="{h - 8}" class="axis" text-anchor="middle">{year}</text>'
        )
    for n, raw in enumerate(series.values()):
        s = raw.dropna()
        if s.empty:
            continue
        step = max(1, len(s) // 900)
        pts = " ".join(
            f"{sx(pd.Timestamp(d)):.1f},{sy(float(v)):.1f}"
            for d, v in list(zip(s.index, s, strict=True))[::step]
        )
        parts.append(f'<polyline points="{pts}" class="s{n}" fill="none"/>')
    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i class="k{n}"></i>{html.escape(name)}</span>'
        for n, name in enumerate(series)
    )
    return f'<div class="legend">{legend}</div>' + "".join(parts)


CSS = """
:root{--bg:#fbfbfa;--fg:#1f2328;--muted:#646b73;--card:#fff;--line:#e5e7ea;--a:#2563eb;--b:#9aa3ad;--neg:#c2410c;--pos:#15803d}
@media (prefers-color-scheme: dark){:root{--bg:#111316;--fg:#e6e8eb;--muted:#9aa3ad;--card:#181b1f;--line:#2a2f35;--a:#60a5fa;--b:#6b7480;--neg:#fb923c;--pos:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.sub{color:var(--muted);margin:0 0 20px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.kpi span{color:var(--muted);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}.neg{color:var(--neg)}.pos{color:var(--pos)}
.chart{width:100%;height:auto}.grid{stroke:var(--line)}.axis{fill:var(--muted);font-size:12px}.s0{stroke:var(--a);stroke-width:2}.s1{stroke:var(--b);stroke-width:1.5}
.legend{display:flex;gap:16px;font-size:13px;color:var(--muted);margin-bottom:4px}.key i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle}
.k0{background:var(--a)}.k1{background:var(--b)}ul{padding-left:20px;color:var(--muted)}code{font-size:13px}
"""


def _cls(v: Any) -> str:
    return (
        "neg"
        if isinstance(v, float) and v < 0
        else ("pos" if isinstance(v, float) and v > 0 else "")
    )


def render_html(
    metrics: dict[str, Any],
    equity: pd.DataFrame,
    benchmark: pd.Series | None,
    *,
    title: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    run_id: str = "",
) -> str:
    idx = growth_index(pd.concat([pd.Series([0.0], index=[equity.index[0]]), twr_returns(equity)]))
    series = {title: idx}
    bm = metrics.get("benchmark") or {}
    bench_name = config.get("benchmark", "benchmark")
    if benchmark is not None and not benchmark.dropna().empty:
        b = benchmark.reindex(equity.index).ffill()
        series[str(bench_name)] = b / b.dropna().iloc[0]
    dd = {title: drawdown(idx)}
    if len(series) > 1:
        dd[str(bench_name)] = drawdown(series[str(bench_name)])

    kpis = [
        ("CAGR (time-weighted)", _pct(metrics.get("cagr")), f"benchmark {_pct(bm.get('cagr'))}"),
        ("After-tax XIRR", _pct(metrics.get("after_tax_xirr")), "if sold at the end"),
        (
            "Max drawdown",
            _pct(metrics.get("max_drawdown")),
            f"benchmark {_pct(bm.get('max_drawdown'))}",
        ),
        ("Sharpe", _num(metrics.get("sharpe")), f"benchmark {_num(bm.get('sharpe'))}"),
        (
            "Turnover / yr",
            _pct(metrics.get("turnover"), 0),
            f"{_num(metrics.get('trades_per_year'), 0)} trades/yr",
        ),
        (
            "Cost + tax drag",
            _pct((metrics.get("cost_drag") or 0) + (metrics.get("tax_drag") or 0), 2),
            f"costs {_pct(metrics.get('cost_drag'), 2)}, tax {_pct(metrics.get('tax_drag'), 2)}",
        ),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><span>{k}</span><b>{v}</b><span>{s}</span></div>' for k, v, s in kpis
    )

    yr = metrics.get("yearly_returns", {})
    byr = bm.get("yearly_returns", {})
    year_rows = "".join(
        f"<tr><td>{y}</td><td class='{_cls(v)}'>{_pct(v)}</td><td class='{_cls(byr.get(y))}'>"
        f"{_pct(byr.get(y))}</td><td class='{_cls(v - byr[y] if y in byr else None)}'>"
        f"{_pct(v - byr[y]) if y in byr else '–'}</td></tr>"
        for y, v in sorted(yr.items())
    )
    detail = [
        ("Period", f"{metrics.get('start')} → {metrics.get('end')} ({metrics.get('years')} yrs)"),
        ("Contributed", _inr(metrics.get("contributions"))),
        ("Final value", _inr(metrics.get("final_value"))),
        ("After tax & exit costs", _inr(metrics.get("after_tax_value"))),
        ("XIRR (pre-tax)", _pct(metrics.get("xirr"))),
        ("Volatility", _pct(metrics.get("volatility"))),
        ("Sortino / Calmar", f"{_num(metrics.get('sortino'))} / {_num(metrics.get('calmar'))}"),
        (
            "Worst drawdown",
            f"{metrics.get('peak')} → {metrics.get('trough')}, "
            f"recovered {metrics.get('recovery') or 'not yet'} ({metrics.get('drawdown_days')} days)",
        ),
        ("Beta / alpha", f"{_num(bm.get('beta'))} / {_pct(bm.get('alpha'))}"),
        (
            "Tracking error / IR",
            f"{_pct(bm.get('tracking_error'))} / {_num(bm.get('information_ratio'))}",
        ),
        ("Avg holding period", f"{_num(metrics.get('avg_holding_days'), 0)} days"),
        ("Sales taxed as long-term", _pct(metrics.get("long_term_share"), 0)),
        ("Hit rate (profitable sales)", _pct(metrics.get("hit_rate"), 0)),
        ("Total costs / taxes", f"{_inr(metrics.get('costs'))} / {_inr(metrics.get('taxes'))}"),
    ]
    for yrs in (1, 3, 5):
        roll = metrics.get(f"rolling_{yrs}y")
        if roll:
            detail.append(
                (
                    f"Rolling {yrs}-yr CAGR",
                    f"worst {_pct(roll['min'])}, median {_pct(roll['median'])}, "
                    f"best {_pct(roll['max'])}",
                )
            )
    detail_rows = "".join(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>" for k, v in detail)
    tax_rows = (
        "".join(
            f"<tr><td>{fy}</td><td>{_inr(v)}</td></tr>"
            for fy, v in metrics.get("taxes_by_fy", {}).items()
            if v
        )
        or "<tr><td colspan=2>No capital-gains tax due</td></tr>"
    )
    cfg_items = ", ".join(f"{k}={v}" for k, v in config.items() if k not in ("start", "end"))
    note_items = "".join(f"<li>{html.escape(n)}</li>" for n in (notes or [])[:30])
    strat = html.escape(json.dumps(strategy_config or {}, indent=1)) if strategy_config else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} backtest</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="sub">Backtest {html.escape(run_id)} · vs {html.escape(str(bench_name))}
· signals at close, fills next open, Indian delivery costs and capital-gains tax</p>
<div class="kpis">{kpi_html}</div>
<h2>Growth of ₹1 (log scale)</h2><div class="card">{_line_chart(series, log=True)}</div>
<h2>Drawdown</h2><div class="card">{_line_chart(dd, height=180, pct_axis=True)}</div>
<h2>Details</h2><div class="card"><table>{detail_rows}</table></div>
<h2>Calendar-year returns</h2><div class="card"><table><tr><th>Year</th><th>Strategy</th>
<th>{html.escape(str(bench_name))}</th><th>Difference</th></tr>{year_rows}</table></div>
<h2>Capital-gains tax paid</h2><div class="card"><table><tr><th>Financial year</th><th>Tax + cess</th></tr>{tax_rows}</table></div>
<h2>Assumptions</h2><div class="card"><code>{html.escape(cfg_items)}</code>
<ul><li>Past performance in a backtest is not a forecast; treat results as hypotheses.</li>
<li>First/partial years use the start date; CAGR is time-weighted, XIRR money-weighted.</li></ul></div>
{f'<h2>Engine notes</h2><div class="card"><ul>{note_items}</ul></div>' if note_items else ""}
{f'<h2>Strategy config</h2><div class="card"><pre>{strat}</pre></div>' if strat else ""}
</main></body></html>"""


def write_reports(
    out_dir: Path,
    metrics: dict[str, Any],
    equity: pd.DataFrame,
    benchmark: pd.Series | None,
    **kw: Any,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    json_path = out_dir / "metrics.json"
    html_path.write_text(render_html(metrics, equity, benchmark, **kw), encoding="utf-8")
    json_path.write_text(json.dumps(to_jsonable(metrics), indent=2), encoding="utf-8")
    return html_path, json_path
