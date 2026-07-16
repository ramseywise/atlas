"""
AtlasState — top-level orchestrator state.

The AtlasAgent receives a user query and routes to one or more of:
  forecast_tool, segment_tool, knowledge_tool
"""
from __future__ import annotations

from typing import Annotated
import operator

from typing_extensions import TypedDict


class ToolCall(TypedDict):
    tool: str           # "forecast" | "segment" | "knowledge"
    args: dict
    result: dict | None
    error: str | None


class AtlasState(TypedDict):
    query: str
    customer_id: str | None
    tool_calls: Annotated[list[ToolCall], operator.add]
    synthesis: str | None   # final LLM-written answer for the user
    error: str | None
