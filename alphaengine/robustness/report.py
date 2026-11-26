"""HTML + JSON output for a robustness analysis."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from alphaengine.backtest.metrics import to_jsonable
from alphaengine.backtest.report import CSS, _line_chart
from alphaengine.robustness.analysis import RobustnessReport, to_dict


def _p(x: Any, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}%}"


def _n(x: Any, d: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _heatmap(metrics: pd.DataFrame, params: list[str], value: str = "sharpe") -> str:
    """Colour-scaled table: rows = first parameter, columns = second (or a single row)."""
    if not params:
        return ""
    rows_p = params[0]
    cols_p = params[1] if len(params) > 1 else None
    vals = metrics[value].astype(float)
    lo, hi = float(vals.min()), float(vals.max())

    def cell(v: float) -> str:
        t = 0.5 if hi == lo else (v - lo) / (hi - lo)
        return (
            f'<td style="background:color-mix(in srgb, var(--a) {int(15 + 70 * t)}%, '
            f'transparent)">{v:.2f}</td>'
        )

    if cols_p is None:
        head = "".join(f"<th>{html.escape(str(v))}</th>" for v in metrics[rows_p])
        body = "".join(cell(float(v)) for v in metrics[value])
        return f"<table><tr><th>{html.escape(rows_p)}</th>{head}</tr><tr><td>{value}</td>{body}</tr></table>"
    pivot = metrics.pivot_table(index=rows_p, columns=cols_p, values=value, aggfunc="mean")
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in pivot.columns)
    body = "".join(
        f"<tr><th>{html.escape(str(i))}</th>" + "".join(cell(float(v)) for v in row) + "</tr>"
        for i, row in pivot.iterrows()
    )
    note = (
        f"<p class='sub'>{value}: rows {html.escape(rows_p)}, columns {html.escape(cols_p)}"
        + (f"; other parameters averaged ({', '.join(params[2:])})" if len(params) > 2 else "")
        + "</p>"
    )
    return f"{note}<table><tr><th></th>{head}</tr>{body}</table>"


def render(r: RobustnessReport) -> str:
    params = list(r.grid)
    sw = r.sweep.metrics
    base = sw.loc[r.base_index]
    o = r.overfitting
    wf = r.walk_forward
    warn = "".join(f"<li>{html.escape(x)}</li>" for x in r.warnings) or (
        "<li>No red flags from these checks. That still doesn't prove the edge is real.</li>"
    )
    kpis = [
        ("Base Sharpe (pre-tax)", _n(base["sharpe"]), f"benchmark {_n(r.benchmark['sharpe'])}"),
        (
            "Deflated Sharpe",
            _p(o["base"].get("deflated_sharpe"), 0),
            f"{o['base'].get('trials')} variants tried",
        ),
        ("PBO", _p(o["pbo"].get("pbo"), 0), "chance the best variant is overfit"),
        (
            "Walk-forward OOS Sharpe",
            _n(wf.get("oos_sharpe")),
            f"in-sample {_n(wf.get('mean_in_sample_sharpe'))}",
        ),
        (
            "CAGR 5th–95th pct",
            f"{_p(r.bootstrap.get('cagr', {}).get('p5'))} – {_p(r.bootstrap.get('cagr', {}).get('p95'))}",
            "bootstrap",
        ),
        (
            "CAGR at 0× / 1× / 2× costs",
            " / ".join(_p(r.costs.get(k)) for k in ("0x", "1x", "2x")),
            "vectorized, pre-tax",
        ),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><span>{k}</span><b>{v}</b><span>{s}</span></div>' for k, v, s in kpis
    )
    cols = [*params, "cagr", "sharpe", "max_drawdown", "turnover", "cost_drag"]
    sweep_rows = "".join(
        "<tr"
        + (" style='font-weight:700'" if i == r.base_index else "")
        + ">"
        + "".join(
            f"<td>{html.escape(str(row[c]))}</td>"
            if c in params
            else f"<td>{_n(row[c]) if c == 'sharpe' else _p(row[c])}</td>"
            for c in cols
        )
        + "</tr>"
        for i, row in sw.sort_values("sharpe", ascending=False).iterrows()
    )
    fold_rows = "".join(
        f"<tr><td>{f['train_end']}</td><td>{f['test_end']}</td><td>{html.escape(str(r.sweep.params[f['chosen']]))}</td>"
        f"<td>{_n(f['in_sample_sharpe'])}</td><td>{_n(f['out_of_sample_sharpe'])}</td></tr>"
        for f in wf.get("folds", [])
    )
    regime_rows = "".join(
        f"<tr><td>{k.replace('_', ' ')}</td><td>{int(v['days'])}</td><td>{_p(v['return_annual'])}</td>"
        f"<td>{_p(v['benchmark_annual'])}</td><td>{_p(v['volatility'])}</td></tr>"
        for k, v in r.regimes.items()
    )
    conf = r.confirmation or {}
    conf_html = (
        (
            f"<table><tr><td>CAGR (vectorized, pre-tax)</td><td>{_p(conf.get('vectorized_cagr'))}</td></tr>"
            f"<tr><td>CAGR (full engine, taxes paid)</td><td>{_p(conf.get('cagr'))}</td></tr>"
            f"<tr><td>After-tax XIRR</td><td>{_p(conf.get('after_tax_xirr'))}</td></tr>"
            f"<tr><td>Max drawdown</td><td>{_p(conf.get('max_drawdown'))}</td></tr>"
            f"<tr><td>Turnover / tax drag / cost drag</td><td>{_p(conf.get('turnover'), 0)} / "
            f"{_p(conf.get('tax_drag'), 2)} / {_p(conf.get('cost_drag'), 2)}</td></tr></table>"
        )
        if conf
        else ""
    )
    curves = {}
    oos = wf.get("oos_returns")
    if isinstance(oos, pd.Series) and not oos.empty:
        curves["walk-forward (out of sample)"] = (1 + oos).cumprod()
        curves["base parameters"] = (
            1 + r.sweep.returns_matrix()[r.base_index].loc[oos.index]
        ).cumprod()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(r.strategy)} robustness</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(r.strategy)}: robustness</h1>
<p class="sub">{html.escape(r.id)} · {len(sw)} variants · grid {html.escape(json.dumps(r.grid))}</p>
<div class="kpis">{kpi_html}</div>
<h2>Warnings</h2><div class="card"><ul>{warn}</ul></div>
<h2>Parameter stability</h2><div class="card">{_heatmap(sw, params)}</div>
<h2>All variants (bold = your config)</h2><div class="card"><table><tr>{"".join(f"<th>{html.escape(c)}</th>" for c in cols)}</tr>{sweep_rows}</table></div>
<h2>Walk-forward</h2><div class="card">{_line_chart(curves, log=True, height=220) if curves else ""}
<table><tr><th>Train to</th><th>Test to</th><th>Chosen</th><th>IS Sharpe</th><th>OOS Sharpe</th></tr>{fold_rows}</table>
<p class="sub">Efficiency (OOS / IS Sharpe): {_n(wf.get("efficiency"))}</p></div>
<h2>Market regimes</h2><div class="card"><table><tr><th>Regime</th><th>Days</th><th>Strategy</th><th>Benchmark</th><th>Volatility</th></tr>{regime_rows}</table></div>
<h2>Full-engine confirmation</h2><div class="card">{conf_html or "not run"}</div>
<h2>How to read this</h2><div class="card"><ul>
<li><b>Deflated Sharpe</b> ≥ 95% means the Sharpe likely isn't luck, even counting every variant tried.</li>
<li><b>PBO</b> under ~30% is reassuring; over 50% means parameter choice is mostly noise.</li>
<li>Prefer a broad plateau in the heatmap over a single bright cell.</li>
<li>Sweeps use the fast vectorized engine (no tax); the confirmation row uses the full engine.</li></ul></div>
</main></body></html>"""


def write(r: RobustnessReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, j = out_dir / "robustness.html", out_dir / "robustness.json"
    h.write_text(render(r), encoding="utf-8")
    j.write_text(json.dumps(to_jsonable(to_dict(r)), indent=2), encoding="utf-8")
    return h, j
