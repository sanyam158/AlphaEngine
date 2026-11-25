import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, unwrap } from "../api/client";
import { Card, ErrorBox, Kpi, Page } from "../components/ui";
import { inr, pct } from "../format";

export default function GoalPage() {
  const runs = useQuery({ queryKey: ["backtests"], queryFn: () => unwrap(api.GET("/api/backtests")) });
  const [form, setForm] = useState({
    target: 20000000, years: 15, current: 1000000, monthly_sip: 50000, step_up: 0.05,
    inflation: 0.06, after_tax: true, success: 0.8, run_id: "", symbol: "NIFTYBEES",
  });
  const plan = useMutation({
    mutationFn: () => unwrap(api.POST("/api/plan/goal", { body: { ...form, run_id: form.run_id || null } })),
  });
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value });
  const r = plan.data;
  const bands = r?.bands.map((b) => ({ ...b, range: [b.p10, b.p90] as [number, number] })) ?? [];

  return (
    <Page title="Goal planner">
      <Card title="Your goal">
        <form className="form" onSubmit={(e) => { e.preventDefault(); plan.mutate(); }}>
          <label>Target (₹, today's money)<input type="number" min={0} step={100000} value={form.target} onChange={set("target")} /></label>
          <label>Years<input type="number" min={1} max={60} value={form.years} onChange={set("years")} /></label>
          <label>Invested now (₹)<input type="number" min={0} step={10000} value={form.current} onChange={set("current")} /></label>
          <label>Monthly SIP (₹)<input type="number" min={0} step={1000} value={form.monthly_sip} onChange={set("monthly_sip")} /></label>
          <label>SIP step-up / yr<input type="number" min={0} max={0.5} step={0.01} value={form.step_up} onChange={set("step_up")} /></label>
          <label>Inflation<input type="number" min={0} max={0.2} step={0.005} value={form.inflation} onChange={set("inflation")} /></label>
          <label>Returns from
            <select value={form.run_id} onChange={set("run_id")}>
              <option value="">Nifty 50 ETF history (NIFTYBEES)</option>
              {runs.data?.map((b) => <option key={b.run_id} value={b.run_id}>{b.strategy} backtest ({pct(b.cagr)})</option>)}
            </select>
          </label>
          <label className="check"><input type="checkbox" checked={form.after_tax} onChange={(e) => setForm({ ...form, after_tax: e.target.checked })} />After exit tax</label>
          <button className="primary" disabled={plan.isPending}>{plan.isPending ? "Simulating…" : "Simulate"}</button>
        </form>
        {plan.error && <ErrorBox error={plan.error} />}
        <p className="muted">5,000 simulated futures built from 12-month blocks of real monthly returns. History doesn't repeat exactly. Treat this as a range, not a promise.</p>
      </Card>
      {r && (
        <>
          <div className="kpis">
            <Kpi label="Chance of reaching it" value={pct(r.probability, 0)} tone={r.probability >= form.success ? "pos" : "neg"}
                 sub={`target ${inr(r.target_nominal)} in ${form.years} yrs`} />
            <Kpi label={`SIP for ${pct(form.success, 0)} confidence`} value={r.required_sip === null ? "–" : `${inr(r.required_sip)}/mo`}
                 sub={form.step_up ? `growing ${pct(form.step_up, 0)}/yr` : undefined} />
            <Kpi label="Typical outcome" value={inr(r.p50)} sub={`bad ${inr(r.p10)} · good ${inr(r.p90)}`} />
            <Kpi label="You put in" value={inr(r.contributed)} sub={r.source} />
          </div>
          <Card title="Range of outcomes by year (10th–90th percentile, median line)">
            <div style={{ width: "100%", height: 280 }}>
              <ResponsiveContainer>
                <AreaChart data={bands} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--line)" vertical={false} />
                  <XAxis dataKey="year" stroke="var(--muted)" fontSize={12} />
                  <YAxis tickFormatter={(v: number) => inr(v)} stroke="var(--muted)" fontSize={12} width={80} />
                  <Tooltip formatter={(v: number | number[]) => Array.isArray(v) ? `${inr(v[0])} – ${inr(v[1])}` : inr(v)}
                           labelFormatter={(y: number) => `Year ${y}`}
                           contentStyle={{ background: "var(--card)", border: "1px solid var(--line)" }} />
                  <Area dataKey="range" stroke="none" fill="var(--series-a)" fillOpacity={0.18} isAnimationActive={false} />
                  <Line dataKey="p50" stroke="var(--series-a)" strokeWidth={2} dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </>
      )}
    </Page>
  );
}
