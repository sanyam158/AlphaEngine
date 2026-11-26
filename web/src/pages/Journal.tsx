import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, unwrap } from "../api/client";
import { Badge, Card, Empty, ErrorBox, Page } from "../components/ui";

export default function JournalPage() {
  const qc = useQueryClient();
  const entries = useQuery({ queryKey: ["journal"], queryFn: () => unwrap(api.GET("/api/journal")) });
  const portfolios = useQuery({ queryKey: ["portfolios"], queryFn: () => unwrap(api.GET("/api/portfolios")) });
  const [form, setForm] = useState({ text: "", portfolio: "", symbol: "", tags: "" });
  const add = useMutation({
    mutationFn: () => unwrap(api.POST("/api/journal", {
      body: { text: form.text, portfolio: form.portfolio || null, symbol: form.symbol || null, tags: form.tags || null },
    })),
    onSuccess: () => {
      setForm({ text: "", portfolio: form.portfolio, symbol: "", tags: "" });
      qc.invalidateQueries({ queryKey: ["journal"] });
    },
  });
  return (
    <Page title="Journal">
      <Card title="Write down why">
        <p className="muted">Record the reasoning behind each decision, and what you expect to happen. Reading it back later is how you learn.</p>
        <form onSubmit={(e) => { e.preventDefault(); add.mutate(); }} style={{ display: "grid", gap: 10 }}>
          <textarea rows={4} required value={form.text} onChange={(e) => setForm({ ...form, text: e.target.value })} aria-label="note" />
          <div className="form">
            <label>Portfolio
              <select value={form.portfolio} onChange={(e) => setForm({ ...form, portfolio: e.target.value })}>
                <option value="">(none)</option>
                {portfolios.data?.map((p) => <option key={p.name}>{p.name}</option>)}
              </select>
            </label>
            <label>Symbol<input value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value })} /></label>
            <label>Tags<input placeholder="rebalance, thesis…" value={form.tags} onChange={(e) => setForm({ ...form, tags: e.target.value })} /></label>
            <button className="primary" disabled={add.isPending}>Add note</button>
          </div>
        </form>
        {add.error && <ErrorBox error={add.error} />}
      </Card>
      <Card title="Entries">
        {entries.error && <ErrorBox error={entries.error} />}
        {entries.data?.length === 0 && <Empty>No notes yet.</Empty>}
        {entries.data?.map((e) => (
          <div key={e.id} style={{ borderBottom: "1px solid var(--line)", padding: "10px 0" }}>
            <div className="muted">
              {e.created_at ? new Date(e.created_at).toLocaleString("en-IN") : ""}{" "}
              {e.portfolio && <Badge>{e.portfolio}</Badge>} {e.symbol && <Badge>{e.symbol}</Badge>} {e.tags && <span>#{e.tags}</span>}
            </div>
            <div style={{ whiteSpace: "pre-wrap" }}>{e.text}</div>
          </div>
        ))}
      </Card>
    </Page>
  );
}
