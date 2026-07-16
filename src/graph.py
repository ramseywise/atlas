"""
AtlasAgent — top-level orchestrator.

Routes user queries to forecast, segment, or knowledge tools.
Uses Claude Haiku to plan tool calls and synthesize the final answer.
"""

from __future__ import annotations

import json
import os

from langgraph.graph import END, StateGraph

from src.state import AtlasState, ToolCall

# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """\
You are Atlas, a financial intelligence assistant. Given a user query, decide which tools to call.

Available tools:
  - forecast: predict future cash flows for a customer (needs customer_id)
  - segment:  discover or refresh customer segments (no args needed)
  - knowledge: look up financial metric definitions or explain concepts (needs query)

User query: {query}
Customer ID (if known): {customer_id}

Respond with valid JSON only:
{{
  "tools": [
    {{"tool": "forecast|segment|knowledge", "args": {{...}}}},
    ...
  ]
}}
Tools may be called in parallel. Use only what's needed.
"""

SYNTHESIS_PROMPT = """\
You are Atlas, a financial intelligence assistant. Synthesize these tool results into a clear,
concise answer for the user. Focus on actionable insights. No raw numbers without context.

User query: {query}

Tool results:
{tool_results}

Write 2-4 sentences max.
"""


def router_node(state: AtlasState) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "tool_calls": [
                ToolCall(tool="knowledge", args={"query": state["query"]}, result=None, error=None)
            ]
        }

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": ROUTER_PROMPT.format(
                    query=state["query"],
                    customer_id=state.get("customer_id") or "unknown",
                ),
            }
        ],
    )
    try:
        data = json.loads(resp.content[0].text.strip())
        calls = [
            ToolCall(tool=t["tool"], args=t.get("args", {}), result=None, error=None)
            for t in data["tools"]
        ]
    except Exception as e:
        calls = [
            ToolCall(tool="knowledge", args={"query": state["query"]}, result=None, error=str(e))
        ]
    return {"tool_calls": calls}


def forecast_tool_node(state: AtlasState) -> dict:
    from src.agents.graph import run_forecasting_agent

    calls = state["tool_calls"]
    updated = []
    for call in calls:
        if call["tool"] == "forecast":
            try:
                result = run_forecasting_agent(call["args"])
                updated.append({**call, "result": result})
            except Exception as e:
                updated.append({**call, "error": str(e)})
        else:
            updated.append(call)
    return {"tool_calls": updated}


def segment_tool_node(state: AtlasState) -> dict:
    calls = state["tool_calls"]
    updated = []
    for call in calls:
        if call["tool"] == "segment":
            try:
                import os
                import tempfile

                from core.preprocessing.synthetic import generate_sequence_dataset
                from src.agents.segment.graph import run_segmentation_agent

                df = generate_sequence_dataset(n_days=365, seed=42)
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                    tmp_path = f.name
                try:
                    df.write_parquet(tmp_path)
                    result = run_segmentation_agent(tmp_path, max_cycles=2, verbose=False)
                    seg = result.get("result") or {}
                    updated.append(
                        {
                            **call,
                            "result": {
                                "n_segments": seg.get("n_segments", 0),
                                "segment_names": seg.get("segment_names", {}),
                            },
                        }
                    )
                finally:
                    os.unlink(tmp_path)
            except Exception as e:
                updated.append({**call, "error": str(e)})
        else:
            updated.append(call)
    return {"tool_calls": updated}


def knowledge_tool_node(state: AtlasState) -> dict:
    from core.knowledge.graph import AtlasGraph

    calls = state["tool_calls"]
    updated = []
    for call in calls:
        if call["tool"] == "knowledge":
            try:
                g = AtlasGraph()
                results = g.search_metrics(call["args"].get("query", ""))
                g.close()
                updated.append({**call, "result": {"metrics": results}})
            except Exception as e:
                updated.append({**call, "error": str(e)})
        else:
            updated.append(call)
    return {"tool_calls": updated}


def synthesizer_node(state: AtlasState) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"synthesis": "API key required for synthesis."}

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    tool_results = json.dumps(
        [
            {"tool": c["tool"], "result": c["result"], "error": c["error"]}
            for c in state["tool_calls"]
        ],
        indent=2,
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": SYNTHESIS_PROMPT.format(
                    query=state["query"],
                    tool_results=tool_results,
                ),
            }
        ],
    )
    return {"synthesis": resp.content[0].text.strip()}


def _route_tools(state: AtlasState) -> str:
    tools = {c["tool"] for c in state["tool_calls"]}
    if "forecast" in tools:
        return "forecast_tool"
    if "segment" in tools:
        return "segment_tool"
    return "knowledge_tool"


def build_atlas_graph() -> StateGraph:
    g = StateGraph(AtlasState)
    g.add_node("router", router_node)
    g.add_node("forecast_tool", forecast_tool_node)
    g.add_node("segment_tool", segment_tool_node)
    g.add_node("knowledge_tool", knowledge_tool_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        _route_tools,
        {
            "forecast_tool": "forecast_tool",
            "segment_tool": "segment_tool",
            "knowledge_tool": "knowledge_tool",
        },
    )
    g.add_edge("forecast_tool", "synthesizer")
    g.add_edge("segment_tool", "synthesizer")
    g.add_edge("knowledge_tool", "synthesizer")
    g.add_edge("synthesizer", END)

    return g.compile()


def run_atlas_agent(query: str, customer_id: str | None = None) -> str:
    graph = build_atlas_graph()
    result = graph.invoke(
        {
            "query": query,
            "customer_id": customer_id,
            "tool_calls": [],
            "synthesis": None,
            "error": None,
        }
    )
    return result.get("synthesis") or "No response generated."
