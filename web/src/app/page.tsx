"use client";

import { useState } from "react";
import { Card, TextInput, Button, Text, Badge } from "@tremor/react";
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

        <Card className="bg-gray-900 border-gray-800">
          <div className="space-y-3">
            <TextInput
              placeholder="Ask anything — 'what is my runway?' or 'explain burn ratio'"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              className="bg-gray-800 border-gray-700 text-white"
            />
            <TextInput
              placeholder="Customer ID (optional)"
              value={customerId}
              onChange={(e) => setCustomerId(e.target.value)}
              className="bg-gray-800 border-gray-700 text-white"
            />
            <Button onClick={handleAsk} loading={loading} className="w-full">
              Ask Atlas
            </Button>
          </div>
        </Card>

        {error && (
          <Card className="bg-red-950 border-red-800">
            <Text className="text-red-300">{error}</Text>
          </Card>
        )}

        {result && (
          <Card className="bg-gray-900 border-gray-800 space-y-3">
            <div className="flex gap-2 flex-wrap">
              {result.tool_calls_made.map((t) => (
                <Badge key={t} color="blue" size="xs">{t}</Badge>
              ))}
            </div>
            <Text className="text-gray-100 leading-relaxed">{result.answer}</Text>
          </Card>
        )}
      </div>
    </main>
  );
}
