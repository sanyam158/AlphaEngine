import type { ReactNode } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SeriesPoint } from "../api/client";
import { shortDate } from "../format";

export function Page({ title, actions, children }: { title: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="page">
      <header className="page-head">
        <h1>{title}</h1>
        {actions && <div className="actions">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

export function Card({ title, children, extra }: { title?: string; children: ReactNode; extra?: ReactNode }) {
  return (
    <div className="card">
      {(title || extra) && (
        <div className="card-head">
          {title && <h2>{title}</h2>}
          {extra}
        </div>
      )}
      {children}
    </div>
  );
}

export function Kpi({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="kpi">
      <span className="kpi-label">{label}</span>
      <b className={tone}>{value}</b>
      {sub && <span className="kpi-sub">{sub}</span>}
    </div>
  );
}

export function Badge({ children, kind = "neutral" }: { children: ReactNode; kind?: string }) {
  return <span className={`badge badge-${kind}`}>{children}</span>;
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <p className="muted">{what}…</p>;
}

export function ErrorBox({ error }: { error: unknown }) {
  return <div className="error">{error instanceof Error ? error.message : String(error)}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

type Series = { name: string; data: SeriesPoint[]; color?: string };

/** Line chart for one or more date series aligned by date. */
export function SeriesChart({
  series,
  height = 260,
  format = (v: number) => v.toFixed(2),
  log = false,
}: {
  series: Series[];
  height?: number;
  format?: (v: number) => string;
  log?: boolean;
}) {
  const byDate = new Map<string, Record<string, number | string>>();
  for (const s of series) {
    for (const p of s.data) {
      const row = byDate.get(p.date) ?? { date: p.date };
      row[s.name] = p.value;
      byDate.set(p.date, row);
    }
  }
  const rows = [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const colors = ["var(--series-a)", "var(--series-b)", "var(--series-c)"];
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="date" tickFormatter={(d: string) => d.slice(0, 4)} minTickGap={40}
                 stroke="var(--muted)" fontSize={12} />
          <YAxis scale={log ? "log" : "auto"} domain={["auto", "auto"]} tickFormatter={format}
                 stroke="var(--muted)" fontSize={12} width={64} allowDataOverflow />
          <Tooltip formatter={(v: number) => format(v)} labelFormatter={(d: string) => shortDate(d)}
                   contentStyle={{ background: "var(--card)", border: "1px solid var(--line)" }} />
          {series.length > 1 && <Legend wrapperStyle={{ fontSize: 13 }} />}
          {series.map((s, i) => (
            <Line key={s.name} dataKey={s.name} dot={false} strokeWidth={i === 0 ? 2 : 1.5}
                  stroke={s.color ?? colors[i % colors.length]} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DrawdownChart({ data, height = 160 }: { data: SeriesPoint[]; height?: number }) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="date" tickFormatter={(d: string) => d.slice(0, 4)} minTickGap={40}
                 stroke="var(--muted)" fontSize={12} />
          <YAxis tickFormatter={(v: number) => `${Math.round(v * 100)}%`} stroke="var(--muted)" fontSize={12} width={64} />
          <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} labelFormatter={(d: string) => shortDate(d)}
                   contentStyle={{ background: "var(--card)", border: "1px solid var(--line)" }} />
          <Area dataKey="value" stroke="var(--neg)" fill="var(--neg)" fillOpacity={0.15} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
