import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, unwrap } from "../api/client";
import { Badge, Card, Empty, ErrorBox, Kpi, Loading, Page } from "../components/ui";
import { inr, pct, tone } from "../format";

export default function HomePage() {
  const portfolios = useQuery({ queryKey: ["portfolios"], queryFn: () => unwrap(api.GET("/api/portfolios")) });
  const health = useQuery({ queryKey: ["data-health"], queryFn: () => unwrap(api.GET("/api/data/health")) });
  const [creating, setCreating] = useState(false);

  const total = (portfolios.data ?? []).reduce((a, p) => a + (p.value ?? p.cash), 0);
  const eq = health.data?.datasets.find((d) => d.dataset === "equities");
  const lastJob = health.data?.jobs[0];

  return (
    <Page title="Home" actions={<button onClick={() => setCreating((c) => !c)}>{creating ? "Close" : "New portfolio"}</button>}>
      <div className="kpis">
        <Kpi label="Paper + tracked value" value={inr(total)} sub={`${portfolios.data?.length ?? 0} portfolio(s)`} />
        <Kpi label="Prices up to" value={eq?.last ?? "–"} sub={eq ? `${eq.symbols.toLocaleString()} symbols` : undefined} />
        <Kpi label="Last daily job" value={lastJob ? lastJob.status : "never"} sub={lastJob ? `${lastJob.run_date}` : "schedule `alphaengine scheduler run-eod`"}
             tone={lastJob?.status === "error" ? "neg" : lastJob?.status === "ok" ? "pos" : ""} />
      </div>
      {creating && <CreatePortfolio onDone={() => setCreating(false)} />}
      <Card title="Portfolios">
        {portfolios.isLoading && <Loading />}
        {portfolios.error && <ErrorBox error={portfolios.error} />}
        {portfolios.data?.length === 0 && <Empty>No portfolios yet. Create a paper portfolio to follow a strategy.</Empty>}
        {!!portfolios.data?.length && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Name</th><th className="text">Strategy</th><th>Type</th><th>Value</th><th>Day</th><th>Since start</th><th>Cash</th><th>Updated</th></tr>
              </thead>
              <tbody>
                {portfolios.data.map((p) => {
                  const value = p.value ?? p.cash;
                  const since = value / p.initial_capital - 1;
                  return (
                    <tr key={p.name}>
                      <td><Link to={`/portfolios/${encodeURIComponent(p.name)}`}>{p.name}</Link>{!p.active && <> <Badge kind="warn">paused</Badge></>}</td>
                      <td className="text">{p.strategy}</td>
                      <td><Badge>{p.broker}</Badge></td>
                      <td>{inr(value)}</td>
                      <td className={tone(p.day_change)}>{pct(p.day_change, 2, true)}</td>
                      <td className={tone(since)}>{pct(since, 1, true)}</td>
                      <td>{inr(p.cash)}</td>
                      <td>{p.last_processed ?? "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </Page>
  );
}

function CreatePortfolio({ onDone }: { onDone: () => void }) {
  const qc = useQueryClient();
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: () => unwrap(api.GET("/api/strategies")) });
  const [form, setForm] = useState({ name: "", strategy: "low_vol_momentum", capital: 1000000, broker: "paper" as "paper" | "manual" });
  const create = useMutation({
    mutationFn: () => unwrap(api.POST("/api/portfolios", { body: { ...form, band: 0, min_trade_value: 2000 } })),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolios"] });
      onDone();
    },
  });
  return (
    <Card title="New portfolio">
      <form className="form" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
        <label>Name<input required pattern="[A-Za-z0-9_\-]{2,64}" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
        <label>Strategy
          <select value={form.strategy} onChange={(e) => setForm({ ...form, strategy: e.target.value })}>
            {strategies.data?.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
          </select>
        </label>
        <label>Capital (₹)<input type="number" min={1000} step={1000} value={form.capital} onChange={(e) => setForm({ ...form, capital: Number(e.target.value) })} /></label>
        <label>Execution
          <select value={form.broker} onChange={(e) => setForm({ ...form, broker: e.target.value as "paper" | "manual" })}>
            <option value="paper">paper (simulated fills)</option>
            <option value="manual">manual (I trade at my broker)</option>
          </select>
        </label>
        <button className="primary" disabled={create.isPending}>Create</button>
      </form>
      {create.error && <ErrorBox error={create.error} />}
      <p className="muted">It rebalances at the next daily job (or use “Preview rebalance” on its page).</p>
    </Card>
  );
}
