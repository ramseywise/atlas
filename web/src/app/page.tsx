"use client";

import { useState } from "react";
import { api, AskResponse } from "@/lib/api";

export default function Home() {
  const [query, setQuery] = useState("");
  const [customerId, setCustomerId] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.ask(query, customerId || undefined);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-950 p-8">
      <div className="max-w-2xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-white">Atlas</h1>
          <p className="text-gray-400 text-sm mt-1">Financial intelligence — forecast · segment · explain</p>
        </div>

        <div className="rounded-lg border bg-gray-900 border-gray-800 p-4">
          <div className="space-y-3">
            <input
              type="text"
              placeholder="Ask anything — 'what is my runway?' or 'explain burn ratio'"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              className="w-full rounded-md bg-gray-800 border border-gray-700 text-white px-3 py-2 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="text"
              placeholder="Customer ID (optional)"
              value={customerId}
              onChange={(e) => setCustomerId(e.target.value)}
              className="w-full rounded-md bg-gray-800 border border-gray-700 text-white px-3 py-2 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleAsk}
              disabled={loading}
              className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Asking…" : "Ask Atlas"}
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-lg border bg-red-950 border-red-800 p-4">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {result && (
          <div className="rounded-lg border bg-gray-900 border-gray-800 p-4 space-y-3">
            <div className="flex gap-2 flex-wrap">
              {result.tool_calls_made.map((t) => (
                <span key={t} className="inline-flex items-center rounded-md bg-blue-500/10 px-2 py-0.5 text-xs font-medium text-blue-400 ring-1 ring-inset ring-blue-500/20">{t}</span>
              ))}
            </div>
            <p className="text-gray-100 leading-relaxed">{result.answer}</p>
          </div>
        )}
      </div>
    </main>
  );
}
