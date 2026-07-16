"""Shared HTML rendering primitives for eval reports."""

from __future__ import annotations


def pct_bar_error(rate: float, width: int = 120, color: str | None = None) -> str:
    """SVG progress bar for error/failure rates — red when high."""
    fill = int(rate * width)
    if color is None:
        color = "#e05252" if rate > 0.5 else "#f0a030" if rate > 0.2 else "#4c9be8"
    return (
        f'<svg width="{width}" height="12" style="vertical-align:middle">'
        f'<rect width="{fill}" height="12" fill="{color}" rx="2"/>'
        f'<rect width="{width}" height="12" fill="none" stroke="#ddd" rx="2"/>'
        f"</svg>"
    )


def pct_bar_pass(rate: float, width: int = 100) -> str:
    """SVG progress bar for pass rates — green when high."""
    fill = max(0, min(width, int(rate * width)))
    color = "#28a745" if rate >= 0.8 else "#ffc107" if rate >= 0.5 else "#dc3545"
    return (
        f'<svg width="{width}" height="10" style="vertical-align:middle;margin-right:6px">'
        f'<rect width="{fill}" height="10" fill="{color}" rx="2"/>'
        f'<rect width="{width}" height="10" fill="none" stroke="#ddd" rx="2"/>'
        f"</svg>"
    )
