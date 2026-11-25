import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { inr, num, pct, tone } from "./format";
import { MetricsTable } from "./pages/BacktestDetail";

describe("format", () => {
  it("formats rupees in lakh and crore", () => {
    expect(inr(1250)).toBe("₹1,250");
    expect(inr(250000)).toBe("₹2.50 L");
    expect(inr(-31500000)).toBe("−₹3.15 Cr");
    expect(inr(null)).toBe("–");
  });
  it("formats percentages and tones", () => {
    expect(pct(0.1234)).toBe("12.3%");
    expect(pct(0.05, 1, true)).toBe("+5.0%");
    expect(pct(-0.02)).toBe("−2.0%");
    expect(pct(Number.NaN)).toBe("–");
    expect(num(1.2345)).toBe("1.23");
    expect(tone(-1)).toBe("neg");
    expect(tone(0)).toBe("");
  });
});

describe("MetricsTable", () => {
  it("renders key metrics with missing values as dashes", () => {
    render(<MetricsTable metrics={{ cagr: 0.15, sharpe: 0.8, benchmark: { beta: 0.9 } }} />);
    expect(screen.getByText("CAGR (time-weighted)")).toBeInTheDocument();
    expect(screen.getByText("15.0%")).toBeInTheDocument();
    expect(screen.getByText("0.80 / –")).toBeInTheDocument();
  });
});

function json(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } }));
}

describe("App", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows portfolios and data freshness on the home page", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.includes("/api/portfolios")) {
        return json([{ name: "lowvol_paper", strategy: "low_vol_momentum", broker: "paper", active: true,
          initial_capital: 1000000, cash: 29564, value: 1010000, last_processed: "2026-11-25", day_change: 0.012 }]);
      }
      if (url.includes("/api/data/health")) {
        return json({ datasets: [{ dataset: "equities", first: "2010-01-04", last: "2026-11-25", trading_days: 4194, symbols: 4208, snapshot: "x" }],
          quality: { run_id: null, counts: {}, examples: [] }, jobs: [{ job: "eod", run_date: "2026-11-25", status: "ok", attempts: 1, message: null, updated_at: null }] });
      }
      return json([]);
    });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(screen.getByText("lowvol_paper")).toBeInTheDocument());
    expect(screen.getByText("+1.0%")).toBeInTheDocument(); // since start
    expect(screen.getByText("+1.20%")).toBeInTheDocument(); // last day
    expect(screen.getAllByText("2026-11-25").length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: "lowvol_paper" })).toHaveAttribute("href", "/portfolios/lowvol_paper");
  });
});
