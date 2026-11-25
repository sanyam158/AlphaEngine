import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { api, unwrap } from "../api/client";
import { Card, DrawdownChart, ErrorBox, Kpi, Loading, Page, SeriesChart } from "../components/ui";
import { inr, num, pct, tone } from "../format";

type M = Record<string, unknown>;
const n = (v: unknown) => (typeof v === "number" ? v : null);

export function MetricsTable({ metrics }: { metrics: M }) {
  const bm = (metrics.benchmark ?? {}) as M;
  const rows: [string, string][] = [
    ["CAGR (time-weighted)", pct(n(metrics.cagr))],
    ["XIRR / after-tax XIRR", `${pct(n(metrics.xirr))} / ${pct(n(metrics.after_tax_xirr))}`],
    ["Volatility", pct(n(metrics.volatility))],
    ["Sharpe / Sortino", `${num(n(metrics.sharpe))} / ${num(n(metrics.sortino))}`],
    ["Max drawdown", `${pct(n(metrics.max_drawdown))} (${metrics.peak ?? "?"} → ${metrics.trough ?? "?"})`],
    ["Beta / alpha", `${num(n(bm.beta))} / ${pct(n(bm.alpha))}`],
    ["Turnover / year", pct(n(metrics.turnover), 0)],
    ["Cost drag / tax drag", `${pct(n(metrics.cost_drag), 2)} / ${pct(n(metrics.tax_drag), 2)}`],
    ["Avg holding period", `${num(n(metrics.avg_holding_days), 0)} days`],
    ["Sales taxed long-term", pct(n(metrics.long_term_share), 0)],
    ["Final / after tax & exit", `${inr(n(metrics.final_value))} / ${inr(n(metrics.after_tax_value))}`],
  ];
  return (
    <table><tbody>{rows.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table>
  );
}

export default function BacktestDetailPage() {
  const { runId = "" } = useParams();
  const q = useQuery({
    queryKey: ["backtest", runId],
    queryFn: () => unwrap(api.GET("/api/backtests/{run_id}", { params: { path: { run_id: runId } } })),
  });
  if (q.isLoading) return <Loading />;
  if (q.error || !q.data) return <ErrorBox error={q.error ?? "not found"} />;
  const d = q.data;
  const m = d.metrics as M;
  const yr = (m.yearly_returns ?? {}) as Record<string, number>;
  const byr = (((m.benchmark ?? {}) as M).yearly_returns ?? {}) as Record<string, number>;
  return (
    <Page title={`${d.strategy} · backtest`}>
      <p className="muted">{d.run_id} · {d.start} → {d.end} · data {d.data_snapshot} · code {d.git_sha?.slice(0, 7) ?? "?"}</p>
      <div className="kpis">
        <Kpi label="CAGR" value={pct(d.cagr)} sub={`benchmark ${pct(d.benchmark_cagr)}`} tone={tone((d.cagr ?? 0) - (d.benchmark_cagr ?? 0))} />
        <Kpi label="After-tax XIRR" value={pct(d.after_tax_xirr)} sub="if sold at the end" />
        <Kpi label="Max drawdown" value={pct(d.max_drawdown)} tone="neg" />
        <Kpi label="Sharpe" value={num(d.sharpe)} />
      </div>
      <Card title="Growth of ₹1 (log scale)">
        <SeriesChart log series={[{ name: d.strategy, data: d.equity }, { name: "benchmark", data: d.benchmark }]} format={(v) => v.toFixed(2)} />
      </Card>
      <Card title="Drawdown"><DrawdownChart data={d.drawdown} /></Card>
      <div className="grid grid-2">
        <Card title="Metrics"><MetricsTable metrics={m} /></Card>
        <Card title="Calendar years">
          <div className="table-wrap"><table>
            <thead><tr><th>Year</th><th>Strategy</th><th>Benchmark</th><th>Diff</th></tr></thead>
            <tbody>{Object.keys(yr).sort().map((y) => (
              <tr key={y}><td>{y}</td><td className={tone(yr[y])}>{pct(yr[y])}</td><td>{pct(byr[y])}</td>
                <td className={tone(yr[y] - (byr[y] ?? 0))}>{byr[y] === undefined ? "–" : pct(yr[y] - byr[y], 1, true)}</td></tr>
            ))}</tbody>
          </table></div>
        </Card>
      </div>
      <Card title="Strategy config"><pre className="code">{JSON.stringify(d.strategy_config, null, 2)}</pre></Card>
    </Page>
  );
}
