import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, unwrap } from "../api/client";
import { Badge, Card, Empty, ErrorBox, Loading, Page } from "../components/ui";
import { pct, num, tone } from "../format";

export default function BacktestsPage() {
  const runs = useQuery({ queryKey: ["backtests"], queryFn: () => unwrap(api.GET("/api/backtests")) });
  return (
    <Page title="Backtest lab">
      <RunForm />
      <Card title="Past runs">
        {runs.isLoading && <Loading />}
        {runs.error && <ErrorBox error={runs.error} />}
        {runs.data?.length === 0 && <Empty>No backtests yet.</Empty>}
        {!!runs.data?.length && (
          <div className="table-wrap"><table>
            <thead><tr><th>Run</th><th className="text">Strategy</th><th>Period</th><th>CAGR</th><th>Benchmark</th><th>After-tax XIRR</th><th>Max DD</th><th>Sharpe</th></tr></thead>
            <tbody>{runs.data.map((r) => (
              <tr key={r.run_id}>
                <td><Link to={`/backtests/${r.run_id}`}>{r.run_id.slice(3, 13)}</Link></td>
                <td className="text">{r.strategy} <span className="muted">v{r.strategy_version}</span></td>
                <td>{r.start?.slice(0, 7)} → {r.end?.slice(0, 7)}</td>
                <td className={tone((r.cagr ?? 0) - (r.benchmark_cagr ?? 0))}>{pct(r.cagr)}</td>
                <td>{pct(r.benchmark_cagr)}</td>
                <td>{pct(r.after_tax_xirr)}</td>
                <td className="neg">{pct(r.max_drawdown)}</td>
                <td>{num(r.sharpe)}</td>
              </tr>
            ))}</tbody>
          </table></div>
        )}
      </Card>
    </Page>
  );
}

function RunForm() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => unwrap(api.GET("/api/strategies")) });
  const [form, setForm] = useState({ strategy: "low_vol_momentum", start: "2015-01-01", capital: 1000000, sip: 0, costs: true, taxes: true });
  const [jobId, setJobId] = useState<string | null>(null);
  const start = useMutation({
    mutationFn: () => unwrap(api.POST("/api/backtests", { body: { ...form, benchmark: "NIFTYBEES" } })),
    onSuccess: (j) => setJobId(j.job_id),
  });
  const job = useQuery({
    queryKey: ["bt-job", jobId],
    enabled: !!jobId,
    queryFn: () => unwrap(api.GET("/api/backtests/jobs/{job_id}", { params: { path: { job_id: jobId! } } })),
    refetchInterval: (q) => (q.state.data && ["done", "error"].includes(q.state.data.status) ? false : 1500),
  });
  useEffect(() => {
    if (job.data?.status === "done" && job.data.run_id) {
      qc.invalidateQueries({ queryKey: ["backtests"] });
      nav(`/backtests/${job.data.run_id}`);
    }
  }, [job.data, nav, qc]);
  const running = !!jobId && job.data?.status !== "done" && job.data?.status !== "error";
  return (
    <Card title="Run a backtest">
      <form className="form" onSubmit={(e) => { e.preventDefault(); setJobId(null); start.mutate(); }}>
        <label>Strategy
          <select value={form.strategy} onChange={(e) => setForm({ ...form, strategy: e.target.value })}>
            {strategies.data?.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
          </select>
        </label>
        <label>Start<input type="date" value={form.start} onChange={(e) => setForm({ ...form, start: e.target.value })} /></label>
        <label>Capital (₹)<input type="number" min={10000} step={10000} value={form.capital} onChange={(e) => setForm({ ...form, capital: Number(e.target.value) })} /></label>
        <label>Monthly SIP (₹)<input type="number" min={0} step={1000} value={form.sip} onChange={(e) => setForm({ ...form, sip: Number(e.target.value) })} /></label>
        <label className="check"><input type="checkbox" checked={form.costs} onChange={(e) => setForm({ ...form, costs: e.target.checked })} />Costs</label>
        <label className="check"><input type="checkbox" checked={form.taxes} onChange={(e) => setForm({ ...form, taxes: e.target.checked })} />Taxes</label>
        <button className="primary" disabled={start.isPending || running}>{running ? "Running…" : "Run"}</button>
      </form>
      {running && <p className="muted"><Badge kind="warn">{job.data?.status ?? "queued"}</Badge> Stock strategies take a minute or two (data load + daily simulation).</p>}
      {start.error && <ErrorBox error={start.error} />}
      {job.data?.status === "error" && <ErrorBox error={job.data.error} />}
    </Card>
  );
}
