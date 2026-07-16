const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface AskResponse {
  answer: string;
  tool_calls_made: string[];
}

export interface ForecastResponse {
  customer_id: string;
  horizon_days: number;
  forecast: number[];
  passed: boolean;
}

export interface Segment {
  id: number;
  label: string;
  description: string;
}

export interface MetricResponse {
  name: string;
  definition: string;
  formula: string | null;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

export const api = {
  ask: (query: string, customer_id?: string) =>
    post<AskResponse>("/ask", { query, customer_id }),

  forecast: (customer_id: string, horizon_days = 30) =>
    post<ForecastResponse>(`/forecast/${customer_id}`, { horizon_days }),

  segments: () =>
    get<{ segments: Segment[] }>("/segments"),

  metric: (name: string) =>
    get<MetricResponse>(`/knowledge/metric?name=${encodeURIComponent(name)}`),
};
