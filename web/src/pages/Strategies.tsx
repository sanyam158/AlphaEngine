import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, unwrap } from "../api/client";
import { Badge, Card, ErrorBox, Loading, Page } from "../components/ui";
import { pct } from "../format";

export default function StrategiesPage() {
  const list = useQuery({ queryKey: ["strategies"], queryFn: () => unwrap(api.GET("/api/strategies")) });
  const [selected, setSelected] = useState<string>("low_vol_momentum");
  return (
    <Page title="Strategies">
      <div className="grid grid-2">
        <Card title="Available">
          {list.isLoading && <Loading />}
          {list.error && <ErrorBox error={list.error} />}
          <div className="table-wrap"><table>
            <thead><tr><th>Name</th><th>Schedule</th><th className="text">Description</th></tr></thead>
            <tbody>{list.data?.map((s) => (
              <tr key={s.name} onClick={() => setSelected(s.name)} style={{ cursor: "pointer", fontWeight: s.name === selected ? 700 : 400 }}>
                <td>{s.name} {s.source === "registered" && <Badge>registered</Badge>}</td>
                <td>{s.schedule}</td>
                <td className="text muted">{s.description}</td>
              </tr>
            ))}</tbody>
          </table></div>
        </Card>
        <Signals name={selected} />
      </div>
      <Editor name={selected} />
    </Page>
  );
}

function Signals({ name }: { name: string }) {
  const q = useQuery({
    queryKey: ["signals", name],
    queryFn: () => unwrap(api.GET("/api/strategies/{name}/signals", { params: { path: { name } } })),
  });
  return (
    <Card title={`Today's target · ${name}`}>
      {q.isLoading && <Loading what="Computing signals" />}
      {q.error && <ErrorBox error={q.error} />}
      {q.data && (
        <>
          <p className="muted">Signals at close {q.data.as_of} · cash {pct(q.data.cash)}</p>
          {q.data.notes.length > 0 && <ul className="notes">{q.data.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
          <div className="table-wrap" style={{ maxHeight: 360, overflowY: "auto" }}><table>
            <thead><tr><th>Symbol</th><th>Weight</th><th className="text">Why</th></tr></thead>
            <tbody>{q.data.positions.map((p) => (
              <tr key={p.symbol}><td>{p.symbol}</td><td>{pct(p.weight)}</td><td className="text muted">{p.reasons.join("; ")}</td></tr>
            ))}</tbody>
          </table></div>
        </>
      )}
    </Card>
  );
}

function Editor({ name }: { name: string }) {
  const detail = useQuery({
    queryKey: ["strategy", name],
    queryFn: () => unwrap(api.GET("/api/strategies/{name}", { params: { path: { name } } })),
  });
  const [text, setText] = useState("");
  useEffect(() => { if (detail.data) setText(detail.data.yaml); }, [detail.data]);
  const validate = useMutation({
    mutationFn: () => unwrap(api.POST("/api/strategies/validate", { body: { text } })),
  });
  return (
    <Card title="Config (edit to experiment, then validate)"
          extra={<button onClick={() => validate.mutate()} disabled={validate.isPending}>Validate</button>}>
      <textarea rows={18} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} aria-label="strategy config" />
      {validate.data?.ok && <p className="pos">Valid: {validate.data.name} (hash {validate.data.config_hash}). Save it as a YAML file and run it from the CLI or the Backtest lab.</p>}
      {validate.data && !validate.data.ok && <ErrorBox error={validate.data.error} />}
      {validate.error && <ErrorBox error={validate.error} />}
    </Card>
  );
}
