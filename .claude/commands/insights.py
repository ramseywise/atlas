#!/usr/bin/env python3
"""
Claude Code Insights — reads ~/.claude/projects/**/*.jsonl, extracts session
stats, calls the Anthropic API once to generate a qualitative HTML report,
and writes it to ~/.claude/usage-data/report.html.

Requires: ANTHROPIC_API_KEY env var (or pass --key <key>)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def iter_sessions(projects_dir: Path) -> list[dict]:
    """Return one dict of stats per .jsonl session file."""
    sessions = []
    for jsonl in sorted(projects_dir.rglob("*.jsonl")):
        if "/subagents/" in str(jsonl):
            continue
        s = parse_session(jsonl)
        if s:
            sessions.append(s)
    return sessions


def _text(content) -> str:
    """Extract plain text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
        return " ".join(parts)
    return ""


def parse_session(path: Path) -> dict | None:
    lines = path.read_text(errors="replace").splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    user_msgs = [r for r in records if r.get("type") == "user"]
    asst_msgs = [r for r in records if r.get("type") == "assistant"]

    if not user_msgs:
        return None

    # Timestamps
    def ts(r):
        t = r.get("timestamp", "")
        if t:
            try:
                return datetime.fromisoformat(t.replace("Z", "+00:00"))
            except ValueError:
                pass
        return None

    user_times = [ts(r) for r in user_msgs if ts(r)]
    if not user_times:
        return None

    start = min(user_times)
    end = max(user_times)
    duration_minutes = (end - start).total_seconds() / 60

    # First real user prompt (skip ide_opened_file injections)
    first_prompt = ""
    for r in user_msgs:
        txt = _text(r.get("message", {}).get("content", ""))
        # strip system injection tags
        txt = re.sub(r"<[^>]+>.*?</[^>]+>", "", txt, flags=re.DOTALL).strip()
        if txt:
            first_prompt = txt[:200]
            break

    # Tool counts from assistant messages
    tool_counts: dict[str, int] = defaultdict(int)
    tool_errors: dict[str, int] = defaultdict(int)
    for r in asst_msgs:
        for block in r.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_counts[block.get("name", "unknown")] += 1

    # Tool result errors from user messages (tool results with is_error)
    for r in user_msgs:
        for block in r.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if block.get("is_error"):
                    # try to categorise
                    content_text = _text(block.get("content", "")).lower()
                    if "no such file" in content_text or "not found" in content_text:
                        tool_errors["file_not_found"] += 1
                    elif "permission" in content_text:
                        tool_errors["permission_denied"] += 1
                    elif "rejected" in content_text:
                        tool_errors["user_rejected"] += 1
                    elif "failed" in content_text or "exit code" in content_text:
                        tool_errors["command_failed"] += 1
                    elif "too large" in content_text or "too long" in content_text:
                        tool_errors["file_too_large"] += 1
                    elif "edit" in content_text:
                        tool_errors["edit_failed"] += 1
                    else:
                        tool_errors["other"] += 1

    # Token usage
    input_tokens = 0
    output_tokens = 0
    for r in asst_msgs:
        usage = r.get("message", {}).get("usage", {})
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

    # Lines added/removed via file-history-snapshots
    files_modified: set[str] = set()
    for r in records:
        if r.get("type") == "file-history-snapshot":
            fid = r.get("fileId", "")
            if fid:
                files_modified.add(fid)

    # Languages (from file extensions touched via tool_use Read/Write/Edit)
    lang_map = {
        "py": "Python",
        "ts": "TypeScript",
        "tsx": "TypeScript",
        "js": "JavaScript",
        "jsx": "JavaScript",
        "md": "Markdown",
        "json": "JSON",
        "yaml": "YAML",
        "yml": "YAML",
        "sh": "Shell",
        "bash": "Shell",
        "ipynb": "Notebook",
        "sql": "SQL",
        "toml": "TOML",
        "html": "HTML",
        "css": "CSS",
    }
    langs: dict[str, int] = defaultdict(int)
    for r in asst_msgs:
        for block in r.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                inp = block.get("input", {})
                fpath = inp.get("file_path", inp.get("path", ""))
                if fpath:
                    ext = Path(fpath).suffix.lstrip(".").lower()
                    lang = lang_map.get(ext)
                    if lang:
                        langs[lang] += 1

    # User response times (gap between consecutive user messages)
    response_times = []
    sorted_user_times = sorted(user_times)
    for i in range(1, len(sorted_user_times)):
        delta = (sorted_user_times[i] - sorted_user_times[i - 1]).total_seconds()
        if 1 < delta < 7200:  # ignore tiny/huge gaps
            response_times.append(delta)

    # Message hours (UTC)
    message_hours = [t.hour for t in user_times]

    # User interruptions (user message mid-tool-use — heuristic: short gap < 5s)
    interruptions = sum(1 for t in response_times if t < 5)

    return {
        "session_id": path.stem,
        "project_path": str(records[0].get("cwd", "")) if records else "",
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_minutes": round(duration_minutes, 1),
        "user_message_count": len(user_msgs),
        "assistant_message_count": len(asst_msgs),
        "tool_counts": dict(tool_counts),
        "tool_errors": dict(tool_errors),
        "languages": dict(langs),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "files_modified": len(files_modified),
        "first_prompt": first_prompt,
        "user_response_times": response_times,
        "user_interruptions": interruptions,
        "message_hours": message_hours,
        "uses_task_agent": tool_counts.get("Task", 0) > 0,
    }


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------


def aggregate(sessions: list[dict]) -> dict:
    if not sessions:
        return {}

    all_tools: dict[str, int] = defaultdict(int)
    all_langs: dict[str, int] = defaultdict(int)
    all_errors: dict[str, int] = defaultdict(int)
    all_hours: list[int] = []
    all_response_times: list[float] = []
    total_user_msgs = 0
    total_files = 0
    total_interruptions = 0
    dates = []

    for s in sessions:
        for k, v in s["tool_counts"].items():
            all_tools[k] += v
        for k, v in s["languages"].items():
            all_langs[k] += v
        for k, v in s["tool_errors"].items():
            all_errors[k] += v
        all_hours.extend(s["message_hours"])
        all_response_times.extend(s["user_response_times"])
        total_user_msgs += s["user_message_count"]
        total_files += s["files_modified"]
        total_interruptions += s["user_interruptions"]
        dates.append(s["start_time"][:10])

    dates.sort()

    # Detect parallel sessions (overlap heuristic)
    sorted_sessions = sorted(sessions, key=lambda x: x["start_time"])
    overlap_events = 0
    sessions_involved: set[str] = set()
    for i, a in enumerate(sorted_sessions):
        for b in sorted_sessions[i + 1 :]:
            if b["start_time"] > a["end_time"]:
                break
            overlap_events += 1
            sessions_involved.add(a["session_id"])
            sessions_involved.add(b["session_id"])

    # Messages during overlaps (rough)
    overlap_messages = sum(
        s["user_message_count"] for s in sessions if s["session_id"] in sessions_involved
    )
    overlap_pct = round(overlap_messages / total_user_msgs * 100) if total_user_msgs else 0

    # Time-of-day buckets
    hour_counts: dict[int, int] = defaultdict(int)
    for h in all_hours:
        hour_counts[h] += 1

    median_rt = (
        sorted(all_response_times)[len(all_response_times) // 2] if all_response_times else 0
    )
    avg_rt = sum(all_response_times) / len(all_response_times) if all_response_times else 0

    # Response time distribution
    rt_buckets = {
        "2-10s": 0,
        "10-30s": 0,
        "30s-1m": 0,
        "1-2m": 0,
        "2-5m": 0,
        "5-15m": 0,
        ">15m": 0,
    }
    for t in all_response_times:
        if t < 10:
            rt_buckets["2-10s"] += 1
        elif t < 30:
            rt_buckets["10-30s"] += 1
        elif t < 60:
            rt_buckets["30s-1m"] += 1
        elif t < 120:
            rt_buckets["1-2m"] += 1
        elif t < 300:
            rt_buckets["2-5m"] += 1
        elif t < 900:
            rt_buckets["5-15m"] += 1
        else:
            rt_buckets[">15m"] += 1

    days_active = len(set(dates))
    msgs_per_day = round(total_user_msgs / days_active, 1) if days_active else 0

    return {
        "session_count": len(sessions),
        "total_user_messages": total_user_msgs,
        "days_active": days_active,
        "msgs_per_day": msgs_per_day,
        "date_range": f"{dates[0]} to {dates[-1]}" if dates else "",
        "top_tools": dict(sorted(all_tools.items(), key=lambda x: -x[1])[:8]),
        "languages": dict(sorted(all_langs.items(), key=lambda x: -x[1])[:6]),
        "tool_errors": dict(sorted(all_errors.items(), key=lambda x: -x[1])[:6]),
        "hour_counts": dict(hour_counts),
        "response_time_distribution": rt_buckets,
        "median_response_time": round(median_rt, 1),
        "avg_response_time": round(avg_rt, 1),
        "total_files_touched": total_files,
        "total_interruptions": total_interruptions,
        "parallel_sessions": {
            "overlap_events": overlap_events,
            "sessions_involved": len(sessions_involved),
            "pct_messages": overlap_pct,
        },
        "uses_task_agent": sum(1 for s in sessions if s["uses_task_agent"]),
    }


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_claude(api_key: str, prompt: str, model: str = "claude-opus-4-5") -> str:
    payload = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        result = json.loads(resp.read())
    return result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are analyzing a developer's Claude Code usage data. Generate a rich, insightful HTML report.

Return ONLY valid HTML starting with <!DOCTYPE html> — no markdown fences, no commentary outside the HTML.

The report must include these sections with these exact id attributes:
- #section-work   : What You Work On (project areas with session counts)
- #section-usage  : How You Use Claude Code (narrative + key insight)
- #section-wins   : Impressive Things You Did (3 big wins)
- #section-friction : Where Things Go Wrong (top 3 friction patterns with examples)
- #section-features : Existing CC Features to Try (3 features with copyable prompts)
- #section-patterns : New Ways to Use Claude Code (3 usage patterns)
- #section-horizon  : On the Horizon (2-3 ambitious future workflows)

Style requirements (inline CSS, no external deps except Google Fonts):
- Clean, modern design using Inter font
- Stat cards at the top showing key numbers
- Bar charts built from divs (no canvas/svg needed)
- Color-coded sections (green for wins, red for friction, blue for features, purple for horizon)
- "At a Glance" summary box at the top with 4 bullet points linking to sections
- A fun/memorable quote at the bottom about something funny or notable from the sessions
- All charts use relative widths (100% = max value)
- Copyable code blocks with copy buttons using clipboard API

Make the analysis specific and personal — use actual session details, first prompts, and patterns
from the data. Avoid generic advice. Reference specific numbers from the stats.
"""


def build_prompt(sessions: list[dict], agg: dict) -> str:
    # Sample of first prompts for context
    sample_prompts = [s["first_prompt"] for s in sessions if s["first_prompt"]][:30]

    # Session summaries (condensed)
    session_summaries = []
    for s in sessions:
        top_tools = sorted(s["tool_counts"].items(), key=lambda x: -x[1])[:3]
        session_summaries.append(
            {
                "id": s["session_id"][:8],
                "date": s["start_time"][:10],
                "duration_min": s["duration_minutes"],
                "user_msgs": s["user_message_count"],
                "first_prompt": s["first_prompt"][:120],
                "top_tools": dict(top_tools),
                "langs": list(s["languages"].keys()),
                "errors": s["tool_errors"],
                "interruptions": s["user_interruptions"],
                "files": s["files_modified"],
            }
        )

    return f"""Analyze this Claude Code usage data and generate the HTML report described in the system prompt.

## Aggregate Statistics
{json.dumps(agg, indent=2)}

## All Session Summaries ({len(session_summaries)} sessions)
{json.dumps(session_summaries, indent=2)}

## Sample of First Prompts (for qualitative analysis)
{json.dumps(sample_prompts, indent=2)}

Generate the complete HTML report now. Remember: return ONLY the HTML, starting with <!DOCTYPE html>.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Claude Code Insights report")
    parser.add_argument("--key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--output", default="~/.claude/usage-data/report.html")
    parser.add_argument("--projects-dir", default="~/.claude/projects")
    parser.add_argument("--model", default="claude-opus-4-5")
    parser.add_argument("--dry-run", action="store_true", help="Extract stats only, skip API call")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: ANTHROPIC_API_KEY not set. Pass --key or set the env var.", file=sys.stderr)
        sys.exit(1)

    projects_dir = Path(args.projects_dir).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning sessions in {projects_dir}...")
    sessions = iter_sessions(projects_dir)
    print(f"Found {len(sessions)} sessions")

    if not sessions:
        print("No sessions found.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(sessions)
    print(f"Date range: {agg['date_range']}")
    print(f"Total user messages: {agg['total_user_messages']}")

    if args.dry_run:
        print("\n--- Aggregate stats ---")
        print(json.dumps(agg, indent=2))
        return

    print(f"Calling Claude API ({args.model}) to generate report...")
    prompt = build_prompt(sessions, agg)

    try:
        html = call_claude(api_key, prompt, model=args.model)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"API error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)

    # Ensure it starts with <!DOCTYPE
    if "<!DOCTYPE" not in html[:100]:
        # Try to extract HTML if model wrapped it
        match = re.search(r"<!DOCTYPE.*", html, re.DOTALL | re.IGNORECASE)
        if match:
            html = match.group(0)
        else:
            print("Warning: response doesn't look like HTML, saving anyway.", file=sys.stderr)

    output_path.write_text(html)
    print(f"\nReport written to: {output_path}")
    print(f"Open with: open {output_path}")


if __name__ == "__main__":
    main()
