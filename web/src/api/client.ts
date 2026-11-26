import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

/** Typed API client generated from the backend's OpenAPI schema (`npm run gen:api`). */
export const api = createClient<paths>({
  baseUrl: typeof window === "undefined" ? "" : window.location.origin,
  fetch: (req: Request) => globalThis.fetch(req), // resolved per call (mockable, polyfill-friendly)
  headers: (() => {
    const key = localStorage.getItem("alphaengine.apiKey");
    return key ? { "X-API-Key": key } : undefined;
  })(),
});

export type Schemas = components["schemas"];
export type PortfolioSummary = Schemas["PortfolioSummary"];
export type PortfolioDetail = Schemas["PortfolioDetail"];
export type OrderOut = Schemas["OrderOut"];
export type BacktestSummary = Schemas["BacktestSummary"];
export type BacktestDetail = Schemas["BacktestDetail"];
export type StrategySummary = Schemas["StrategySummary"];
export type SeriesPoint = Schemas["SeriesPoint"];
export type RebalanceOut = Schemas["RebalanceOut"];

/** Unwrap an openapi-fetch result, throwing a readable error on failure. */
export async function unwrap<T>(p: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  const { data, error, response } = await p;
  if (error !== undefined || data === undefined) {
    const detail =
      error && typeof error === "object" && "detail" in error ? (error as { detail: unknown }).detail : error;
    throw new Error(typeof detail === "string" ? detail : `HTTP ${response.status}`);
  }
  return data;
}
