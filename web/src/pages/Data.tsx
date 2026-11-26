import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "../api/client";
import { Badge, Card, Empty, ErrorBox, Loading, Page } from "../components/ui";

export default function DataPage() {
  const q = useQuery({ queryKey: ["data-health"], queryFn: () => unwrap(api.GET("/api/data/health")) });
  if (q.isLoading) return <Loading />;
  if (q.error || !q.data) return <ErrorBox error={q.error} />;
  const { datasets, quality, jobs } = q.data;
  return (
    <Page title="Data health">
      <Card title="Datasets">
        <table><thead><tr><th>Dataset</th><th>From</th><th>To</th><th>Trading days</th><th>Symbols</th><th>Snapshot</th></tr></thead>
          <tbody>{datasets.map((d) => (
            <tr key={d.dataset}><td>{d.dataset}</td><td>{d.first ?? "–"}</td><td>{d.last ?? "–"}</td>
              <td>{d.trading_days.toLocaleString()}</td><td>{d.symbols.toLocaleString()}</td><td><code>{d.snapshot}</code></td></tr>
          ))}</tbody></table>
      </Card>
      <div className="grid grid-2">
        <Card title="Latest quality check">
          {!quality.run_id ? <Empty>Run <code>alphaengine data quality</code>.</Empty> : (
            <>
              <p className="muted">{quality.run_id}</p>
              {(["error", "warning", "info"] as const).map((sev) => (
                <p key={sev}><Badge kind={sev === "error" ? "err" : sev === "warning" ? "warn" : "neutral"}>{sev}</Badge>{" "}
                  {Object.entries(quality.counts[sev] ?? {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "none"}</p>
              ))}
              <div className="table-wrap"><table>
                <thead><tr><th>Check</th><th>Symbol</th><th>Date</th><th className="text">Detail</th></tr></thead>
                <tbody>{quality.examples.map((e, i) => (
                  <tr key={i}><td>{String(e.check)}</td><td>{String(e.symbol ?? "")}</td><td>{String(e.date)}</td><td className="text muted">{String(e.detail)}</td></tr>
                ))}</tbody>
              </table></div>
            </>
          )}
        </Card>
        <Card title="Daily job runs">
          {jobs.length === 0 ? <Empty>The daily job hasn't run yet.</Empty> : (
            <table><thead><tr><th>Date</th><th>Status</th><th>Tries</th><th className="text">Message</th></tr></thead>
              <tbody>{jobs.map((j) => (
                <tr key={j.job + j.run_date}><td>{j.run_date}</td>
                  <td><Badge kind={j.status === "ok" ? "ok" : j.status === "error" ? "err" : "warn"}>{j.status}</Badge></td>
                  <td>{j.attempts}</td><td className="text muted">{j.message}</td></tr>
              ))}</tbody></table>
          )}
        </Card>
      </div>
    </Page>
  );
}
