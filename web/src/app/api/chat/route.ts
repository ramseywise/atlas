import { anthropic } from "@ai-sdk/anthropic";
import { streamText } from "ai";

// POST /api/chat — minimal streamText endpoint showing Vercel AI SDK integration pattern.
// Streams Claude's response chunk-by-chunk; the frontend can consume this with useChat
// (from 'ai/react') pointed at this route.
//
// Usage (frontend):
//   import { useChat } from 'ai/react';
//   const { messages, input, handleInputChange, handleSubmit } = useChat({ api: '/api/chat' });
//
// To wire into the existing Atlas search-synthesize flow, replace the static
// `systemPrompt` below with atlas context and pass query/customer_id via `messages`.

export const runtime = "edge";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: anthropic("claude-sonnet-4-5"),
    system:
      "You are Atlas, a financial intelligence assistant. " +
      "Answer questions about cash flow, customer segments, and business metrics concisely.",
    messages,
  });

  return result.toDataStreamResponse();
}
