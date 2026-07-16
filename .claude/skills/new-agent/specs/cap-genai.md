# cap-genai — Templates for genai-tools-builder subagent

## File: {OUTPUT_DIR}/tool_registry.py

```python
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, get_type_hints

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Decorator-based registry for LLM function-calling tools.

    Usage:
        registry = ToolRegistry()

        @registry.tool(name="my_tool", description="Does something useful")
        def my_tool(query: str, limit: int = 10) -> str:
            return f"Results for {query}"

        # Get LLM-compatible tool defs
        tools = registry.all_tools()

        # Dispatch by name
        result = registry.dispatch("my_tool", {"query": "test", "limit": 5})
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def tool(
        self,
        name: str | None = None,
        description: str = "",
    ) -> Callable:
        """Decorator that registers a function as an LLM-callable tool.

        Args:
            name: Tool name visible to the LLM. Defaults to the function's __name__.
            description: Human-readable description of what the tool does.
        """
        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            schema = _build_schema(fn, tool_name, description)
            self._tools[tool_name] = {"fn": fn, "schema": schema}
            logger.debug("Registered tool: %s", tool_name)
            return fn

        return decorator

    def all_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools in LLM function-calling format.

        Compatible with Anthropic tool_use, OpenAI functions, and Gemini function_declarations.
        """
        return [entry["schema"] for entry in self._tools.values()]

    def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        """Call the registered function by name with the given arguments.

        Args:
            name: Registered tool name.
            args: Dict of keyword arguments to pass to the function.

        Returns:
            The function's return value.

        Raises:
            KeyError: If no tool with that name is registered.
            TypeError: If the arguments don't match the function signature.
        """
        if name not in self._tools:
            available = ", ".join(sorted(self._tools))
            raise KeyError(f"Tool {name!r} not registered. Available: {available}")

        fn = self._tools[name]["fn"]
        logger.debug("Dispatching tool %r with args %s", name, list(args))

        # Filter out keys not in the function signature to be safe
        sig = inspect.signature(fn)
        valid_keys = set(sig.parameters)
        filtered_args = {k: v for k, v in args.items() if k in valid_keys}

        return fn(**filtered_args)

    def registered_names(self) -> list[str]:
        return list(self._tools)


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

_PYTHON_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "Any": "string",  # fallback
}


def _build_schema(fn: Callable, name: str, description: str) -> dict[str, Any]:
    """Build an Anthropic/OpenAI-compatible tool schema from a Python function."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        type_hint = hints.get(param_name)
        json_type = _resolve_json_type(type_hint)

        param_schema: dict[str, Any] = {"type": json_type}

        # Extract default description from docstring if present
        if fn.__doc__:
            import re  # noqa: PLC0415
            # Match "param_name: description" in docstring Args section
            pattern = rf"{re.escape(param_name)}\s*:\s*(.+)"
            m = re.search(pattern, fn.__doc__)
            if m:
                param_schema["description"] = m.group(1).strip()

        properties[param_name] = param_schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _resolve_json_type(hint: Any) -> str:
    if hint is None:
        return "string"
    # Handle Optional[X] → unwrap
    origin = getattr(hint, "__origin__", None)
    if origin is type(None):
        return "string"
    if hasattr(hint, "__args__"):
        # Optional[X] has __args__ = (X, NoneType)
        non_none = [a for a in hint.__args__ if a is not type(None)]
        if non_none:
            return _resolve_json_type(non_none[0])
    type_name = getattr(hint, "__name__", str(hint))
    return _PYTHON_TO_JSON_TYPE.get(type_name, "string")


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

registry = ToolRegistry()
```

## File: {OUTPUT_DIR}/shared_tools.py

```python
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from {AGENT_NAME}.tool_registry import registry


# ---------------------------------------------------------------------------
# Currency formatter
# ---------------------------------------------------------------------------

@registry.tool(
    name="format_currency",
    description="Format a numeric amount as a currency string (e.g., 1234.5 EUR → '€1,234.50').",
)
def format_currency(amount: float, currency: str = "EUR") -> str:
    """Format amount as a currency string.

    Args:
        amount: Numeric value to format.
        currency: ISO 4217 currency code (e.g., EUR, USD, GBP). Defaults to EUR.
    """
    _SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "CHF ", "DKK": "DKK "}
    symbol = _SYMBOLS.get(currency.upper(), f"{currency.upper()} ")

    if amount < 0:
        formatted = f"-{symbol}{abs(amount):,.2f}"
    else:
        formatted = f"{symbol}{amount:,.2f}"

    return formatted


# ---------------------------------------------------------------------------
# Date range parser
# ---------------------------------------------------------------------------

@registry.tool(
    name="parse_date_range",
    description=(
        "Parse a natural language date range like 'last 30 days', 'this month', "
        "or 'Q1 2024' into a (start_date, end_date) pair."
    ),
)
def parse_date_range(text: str) -> tuple[date, date]:
    """Parse a natural language date range expression.

    Args:
        text: Natural language range (e.g., 'last 7 days', 'this month', 'Q2 2024').

    Returns:
        Tuple of (start_date, end_date) as date objects.
    """
    today = date.today()
    text_lower = text.lower().strip()

    # last N days
    m = re.match(r"last\s+(\d+)\s+days?", text_lower)
    if m:
        n = int(m.group(1))
        return today - timedelta(days=n), today

    # last N weeks
    m = re.match(r"last\s+(\d+)\s+weeks?", text_lower)
    if m:
        n = int(m.group(1))
        return today - timedelta(weeks=n), today

    # last N months (approximated as 30-day blocks)
    m = re.match(r"last\s+(\d+)\s+months?", text_lower)
    if m:
        n = int(m.group(1))
        return today - timedelta(days=n * 30), today

    # this week (Mon–Sun)
    if text_lower in ("this week", "current week"):
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=6)

    # this month
    if text_lower in ("this month", "current month"):
        start = today.replace(day=1)
        # Last day of month
        if today.month == 12:
            end = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(today.year, today.month + 1, 1) - timedelta(days=1)
        return start, end

    # this year
    if text_lower in ("this year", "current year", "ytd"):
        return date(today.year, 1, 1), today

    # Q{N} [YYYY]
    m = re.match(r"q([1-4])(?:\s+(\d{4}))?", text_lower)
    if m:
        q = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else today.year
        quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        start = date(year, *quarter_starts[q])
        end = date(year, *quarter_ends[q])
        return start, end

    # yesterday
    if text_lower == "yesterday":
        return today - timedelta(days=1), today - timedelta(days=1)

    # today
    if text_lower == "today":
        return today, today

    raise ValueError(f"Could not parse date range: {text!r}")


# ---------------------------------------------------------------------------
# Table summariser
# ---------------------------------------------------------------------------

@registry.tool(
    name="summarize_table",
    description=(
        "Summarise a list of dicts (table rows) as a compact markdown table, "
        "truncating to max_rows rows."
    ),
)
def summarize_table(data: list[dict[str, Any]], max_rows: int = 10) -> str:
    """Format a list of dicts as a compact markdown table.

    Args:
        data: List of row dicts (all must share the same keys).
        max_rows: Maximum rows to include before truncation notice.
    """
    if not data:
        return "_No data_"

    headers = list(data[0].keys())
    rows = data[:max_rows]
    truncated = len(data) > max_rows

    def _fmt(val: Any) -> str:
        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    header_row = " | ".join(headers)
    separator = " | ".join("---" for _ in headers)
    data_rows = [" | ".join(_fmt(row.get(h)) for h in headers) for row in rows]

    table = f"| {header_row} |\n| {separator} |\n"
    table += "\n".join(f"| {row} |" for row in data_rows)

    if truncated:
        table += f"\n\n_Showing {max_rows} of {len(data)} rows._"

    return table
```
