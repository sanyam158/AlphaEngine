# Dashboard and API (Phase 7)

```bash
cd web && npm install && npm run build && cd ..   # once, and after frontend changes
alphaengine serve                                 # http://127.0.0.1:8000
```

| Page | What it does |
|---|---|
| **Home** | Total value, how fresh the data is, last daily-job status, every portfolio's day and since-start returns. Create portfolios here. |
| **Portfolio** | Value over time, holdings with P&L and how many shares are long-term, open and recent orders, a capital-gains estimate. **Preview rebalance** shows the trade list, tax notes and risk findings. For manual portfolios you record fills, with charges, row by row. **Pause** is the kill switch. |
| **Backtest lab** | Run any built-in strategy (in the background) with costs, taxes and SIP toggles. Browse past runs. A run's page shows growth vs benchmark (log scale), drawdown, metrics and calendar years. |
| **Strategies** | Today's target for any strategy with the reasons, plus a config editor that validates YAML or JSON. |
| **Journal** | A log of why you made each decision, tagged by portfolio and symbol. |
| **Data health** | Dataset ranges and snapshot ids, the latest quality check, and the daily-job history. |

## API
- JSON under `/api`, documented at `http://127.0.0.1:8000/docs` (OpenAPI).
- **Access:** only from this machine by default. To expose it (for example on your LAN), set
  `ALPHAENGINE_API_KEY`, start with `alphaengine serve --host 0.0.0.0`, and store the key in the
  browser with `localStorage.setItem("alphaengine.apiKey", "...")`.
- **Backtests** run in a background worker, one at a time because of memory:
  `POST /api/backtests`, then poll `GET /api/backtests/jobs/{id}`.

## Frontend development
```bash
alphaengine serve            # API on :8000
cd web && npm run dev        # Vite on :5173, proxies /api to :8000
npm test                     # vitest + Testing Library
npm run gen:api              # regenerate src/api/schema.d.ts from openapi.json after API changes
```
To refresh `web/openapi.json`, run
`python -c "import json; from alphaengine.api.app import create_app; print(json.dumps(create_app(web_dist=None).openapi()))" > web/openapi.json`.
The UI uses React, TanStack Query, React Router, Recharts, and typed calls through
`openapi-fetch`. Styling is plain CSS with light and dark themes.

## Not done yet
- No Playwright end-to-end tests yet. Pages were checked by hand in a browser. Component and
  API tests run in the suites.
- Recharts is on v2, which upstream no longer maintains. Moving to v3 is a small follow-up.
