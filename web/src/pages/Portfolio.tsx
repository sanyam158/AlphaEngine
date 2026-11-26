import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { api, unwrap, type OrderOut, type RebalanceOut } from "../api/client";
import { Badge, Card, Empty, ErrorBox, Kpi, Loading, Page, SeriesChart } from "../components/ui";
import { inr, num, pct, tone } from "../format";

export default function PortfolioPage() {
  const { name = "" } = useParams();
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["portfolio", name],
    queryFn: () => unwrap(api.GET("/api/portfolios/{name}", { params: { path: { name } } })),
  });
  const tax = useQuery({
    queryKey: ["tax", name],
    queryFn: () => unwrap(api.GET("/api/portfolios/{name}/tax", { params: { path: { name } } })),
  });
  const harvest = useQuery({
    queryKey: ["harvest", name],
    queryFn: () => unwrap(api.GET("/api/portfolios/{name}/harvest", { params: { path: { name } } })),
  });
  const [plan, setPlan] = useState<RebalanceOut | null>(null);
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["portfolio", name] });
    qc.invalidateQueries({ queryKey: ["portfolios"] });
  };
  const rebalance = useMutation({
    mutationFn: (dry: boolean) =>
      unwrap(api.POST("/api/portfolios/{name}/rebalance", { params: { path: { name }, query: { dry_run: dry } } })),
    onSuccess: (r) => { setPlan(r); refresh(); },
  });
  const toggle = useMutation({
    mutationFn: (action: "pause" | "resume") =>
      unwrap(api.POST("/api/portfolios/{name}/{action}", { params: { path: { name, action } } })),
    onSuccess: refresh,
  });

  if (q.isLoading) return <Loading />;
  if (q.error || !q.data) return <ErrorBox error={q.error ?? "not found"} />;
  const p = q.data;
  const value = p.value ?? p.cash;
  const invested = p.holdings.reduce((a, h) => a + h.value, 0);
  const pnl = p.holdings.reduce((a, h) => a + h.pnl, 0);

  return (
    <Page
      title={p.name}
      actions={
        <>
          <Badge>{p.broker}</Badge>
          {!p.active && <Badge kind="warn">paused</Badge>}
          <button onClick={() => rebalance.mutate(true)} disabled={rebalance.isPending}>Preview rebalance</button>
          <button onClick={() => toggle.mutate(p.active ? "pause" : "resume")}>{p.active ? "Pause (kill switch)" : "Resume"}</button>
        </>
      }
    >
      <p className="muted">Follows <b>{p.strategy}</b> · last processed {p.last_processed ?? "never"}</p>
      <div className="kpis">
        <Kpi label="Value" value={inr(value)} sub={`started with ${inr(p.initial_capital)}`} />
        <Kpi label="Since start" value={pct(value / p.initial_capital - 1, 2, true)} tone={tone(value / p.initial_capital - 1)} />
        <Kpi label="Last day" value={pct(p.day_change, 2, true)} tone={tone(p.day_change)} />
        <Kpi label="Invested / cash" value={pct(value ? invested / value : 0, 0)} sub={`cash ${inr(p.cash)}`} />
        <Kpi label="Unrealised P&L" value={inr(pnl)} tone={tone(pnl)} />
      </div>
      {rebalance.error && <ErrorBox error={rebalance.error} />}
      {plan && <PlanCard plan={plan} broker={p.broker} onExecute={() => rebalance.mutate(false)} busy={rebalance.isPending} />}
      <Card title="Value over time">
        {p.equity.length > 1 ? <SeriesChart series={[{ name: "value", data: p.equity }]} format={(v) => inr(v)} />
          : <Empty>The chart fills in as the daily job runs.</Empty>}
      </Card>
      <div className="grid grid-2">
        <Card title={`Holdings (${p.holdings.length})`}>
          {p.holdings.length === 0 ? <Empty>No holdings yet.</Empty> : (
            <div className="table-wrap"><table>
              <thead><tr><th>Symbol</th><th>Qty</th><th>Price</th><th>Value</th><th>Weight</th><th>P&L</th><th title="Shares that would be taxed as long-term if sold today">LT qty</th></tr></thead>
              <tbody>{p.holdings.map((h) => (
                <tr key={h.symbol}><td>{h.symbol}</td><td>{h.quantity}</td><td>{num(h.price)}</td><td>{inr(h.value)}</td>
                  <td>{pct(h.weight)}</td><td className={tone(h.pnl)}>{inr(h.pnl)}</td><td>{h.long_term_qty}</td></tr>
              ))}</tbody>
            </table></div>
          )}
        </Card>
        <Card title={`Open orders (${p.open_orders.length})`}>
          {p.open_orders.length === 0 ? <Empty>Nothing pending.</Empty> : <Orders orders={p.open_orders} manual={p.broker === "manual"} onChange={refresh} />}
        </Card>
      </div>
      <div className="grid grid-2">
        <Card title="Recent orders">
          <div className="table-wrap"><table>
            <thead><tr><th>#</th><th>Signal</th><th>Side</th><th>Symbol</th><th>Qty</th><th>Fill</th><th>Status</th></tr></thead>
            <tbody>{p.recent_orders.slice(0, 30).map((o) => (
              <tr key={o.id}><td>{o.id}</td><td>{o.signal_date}</td><td>{o.side}</td><td>{o.symbol}</td><td>{o.filled_quantity}/{o.quantity}</td>
                <td>{o.fill_price ? num(o.fill_price) : "–"}</td><td><StatusBadge s={o.status} /></td></tr>
            ))}</tbody>
          </table></div>
        </Card>
        <Card title="Harvesting ideas (this FY)">
          {harvest.data?.length ? (
            <div className="table-wrap"><table>
              <thead><tr><th>Type</th><th>Symbol</th><th>Qty</th><th>Gain</th><th>Tax now</th><th>Costs</th><th className="text">Why</th></tr></thead>
              <tbody>{harvest.data.map((h) => (
                <tr key={h.kind + h.symbol}><td>{h.kind === "loss" ? "loss" : "LTCG"}</td><td>{h.symbol}</td><td>{h.quantity}</td>
                  <td className={tone(h.gain)}>{inr(h.gain)}</td><td className={tone(-h.tax_change)}>{inr(h.tax_change)}</td>
                  <td>{inr(h.round_trip_cost)}</td><td className="text muted">{h.note}</td></tr>
              ))}</tbody>
            </table></div>
          ) : <Empty>Nothing worth harvesting right now.</Empty>}
          <p className="muted">Sell and buy back to keep exposure. Check large amounts with a tax professional.</p>
        </Card>
        <Card title="Capital-gains tax (estimate)">
          {tax.data?.length ? (
            <table><thead><tr><th>FY</th><th>STCG</th><th>LTCG</th><th>Tax + cess</th></tr></thead>
              <tbody>{tax.data.map((t) => <tr key={t.fy}><td>{t.fy}</td><td>{inr(t.stcg)}</td><td>{inr(t.ltcg)}</td><td>{inr(t.tax)}</td></tr>)}</tbody></table>
          ) : <Empty>No realised sales yet.</Empty>}
        </Card>
      </div>
    </Page>
  );
}

function StatusBadge({ s }: { s: string }) {
  const kind = s === "FILLED" ? "ok" : s === "CANCELLED" || s === "REJECTED" ? "err" : "warn";
  return <Badge kind={kind}>{s.toLowerCase()}</Badge>;
}

function PlanCard({ plan, broker, onExecute, busy }: { plan: RebalanceOut; broker: string; onExecute: () => void; busy: boolean }) {
  const blocked = plan.findings.some((f) => f.level === "error");
  return (
    <Card title={`Rebalance ${plan.orders_created ? "orders created" : "preview"} · signals at close ${plan.as_of}`}
          extra={!plan.orders_created && plan.trades.length > 0 && (
            <button className="primary" onClick={onExecute} disabled={busy || blocked}>
              {broker === "paper" ? "Create paper orders" : "Create my order checklist"}
            </button>
          )}>
      {plan.notes.length > 0 && <ul className="notes">{plan.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
      {plan.findings.map((f) => <p key={f.message} className={f.level === "error" ? "neg" : "muted"}>[{f.level}] {f.message}</p>)}
      {plan.trades.length === 0 && plan.deferred.length === 0 ? <Empty>Already on target: nothing to trade.</Empty> : (
        <div className="table-wrap"><table>
          <thead><tr><th>Side</th><th>Symbol</th><th>Qty</th><th>~Price</th><th>Value</th><th className="text">Why</th></tr></thead>
          <tbody>
            {plan.trades.map((t) => <tr key={t.side + t.symbol}><td>{t.side}</td><td>{t.symbol}</td><td>{t.quantity}</td><td>{num(t.est_price)}</td><td>{inr(t.value)}</td><td className="text">{t.reason}{t.note ? ` · ${t.note}` : ""}</td></tr>)}
            {plan.deferred.map((t) => <tr key={"d" + t.symbol} className="muted"><td>WAIT</td><td>{t.symbol}</td><td>{t.quantity}</td><td>{num(t.est_price)}</td><td>{inr(t.value)}</td><td className="text">{t.note}</td></tr>)}
          </tbody>
        </table></div>
      )}
      <p className="muted">Estimated charges {inr(plan.est_charges)}. Paper orders fill at the next trading day's open.</p>
    </Card>
  );
}

function Orders({ orders, manual, onChange }: { orders: OrderOut[]; manual: boolean; onChange: () => void }) {
  return (
    <div className="table-wrap"><table>
      <thead><tr><th>#</th><th>Side</th><th>Symbol</th><th>Qty</th><th>~Price</th><th className="text">{manual ? "Record your fill" : "Status"}</th></tr></thead>
      <tbody>{orders.map((o) => <OrderRow key={o.id} o={o} manual={manual} onChange={onChange} />)}</tbody>
    </table></div>
  );
}

function OrderRow({ o, manual, onChange }: { o: OrderOut; manual: boolean; onChange: () => void }) {
  const left = o.quantity - o.filled_quantity;
  const [qty, setQty] = useState(left);
  const [price, setPrice] = useState(o.est_price);
  const [charges, setCharges] = useState<string>("");
  const today = new Date().toISOString().slice(0, 10);
  const fill = useMutation({
    mutationFn: () => unwrap(api.POST("/api/orders/{order_id}/fill", {
      params: { path: { order_id: o.id } },
      body: { quantity: qty, price, day: today, charges: charges === "" ? null : Number(charges) },
    })),
    onSuccess: onChange,
  });
  const cancel = useMutation({
    mutationFn: () => unwrap(api.POST("/api/orders/{order_id}/cancel", { params: { path: { order_id: o.id } } })),
    onSuccess: onChange,
  });
  return (
    <tr>
      <td>{o.id}</td><td>{o.side}</td><td>{o.symbol}</td><td>{o.filled_quantity ? `${o.filled_quantity}/${o.quantity}` : o.quantity}</td><td>{num(o.est_price)}</td>
      <td className="text">
        {manual ? (
          <span style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            <input aria-label="quantity" type="number" min={1} max={left} value={qty} onChange={(e) => setQty(Number(e.target.value))} style={{ width: 80 }} />
            <input aria-label="price" type="number" step="0.05" value={price} onChange={(e) => setPrice(Number(e.target.value))} style={{ width: 96 }} />
            <input aria-label="charges" placeholder="charges" type="number" step="0.01" value={charges} onChange={(e) => setCharges(e.target.value)} style={{ width: 90 }} />
            <button onClick={() => fill.mutate()} disabled={fill.isPending}>Filled</button>
            <button onClick={() => cancel.mutate()} disabled={cancel.isPending}>Cancel</button>
          </span>
        ) : <><StatusBadge s={o.status} /> {o.note ?? ""}</>}
        {(fill.error || cancel.error) && <ErrorBox error={fill.error ?? cancel.error} />}
      </td>
    </tr>
  );
}
