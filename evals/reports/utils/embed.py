"""Embed exported SVG figures into stats / comparison HTML."""

from __future__ import annotations

import html as _html
import re
from pathlib import Path

CHART_MARKER = "<!-- eval-charts -->"
NARRATIVE_MARKER = "<!-- eval-narrative -->"



def read_svg(fig_dir: Path, name: str) -> str:
    path = fig_dir / f"{name}.svg"
    if not path.exists():
        return f'<p class="chart-note">Missing {name}.svg — run <code>make reports-validate</code></p>'
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<\?xml[^?]*\?>", "", raw).strip()
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw).strip()
    raw = re.sub(r'(<svg[^>]*)\s+width="[^"]*"', r"\1", raw, count=1)
    raw = re.sub(r'(<svg[^>]*)\s+height="[^"]*"', r"\1", raw, count=1)
    return raw


def chart_card(
    title: str,
    note: str,
    svg_name: str,
    fig_dir: Path,
    *,
    wide: bool = False,
    tall: bool = False,
) -> str:
    wide_cls = " wide" if wide else ""
    wrap_cls = " chart-wrap" if tall else ""
    return (
        f'<div class="card{wide_cls}">'
        f"<h3>{_html.escape(title)}</h3>"
        f'<p class="chart-note">{_html.escape(note)}</p>'
        f'<div class="fig-embed{wrap_cls}">{read_svg(fig_dir, svg_name)}</div>'
        f"</div>"
    )


def inject_narrative_section(html_path: Path, section_html: str, *, extra_css: str = "") -> None:
    """Insert or replace narrative block at NARRATIVE_MARKER (top of .page)."""
    _ensure_theme_css(html_path)
    html = html_path.read_text(encoding="utf-8")
    block = f"{NARRATIVE_MARKER}\n{section_html}\n"
    if NARRATIVE_MARKER in html:
        html = re.sub(
            rf"{re.escape(NARRATIVE_MARKER)}\s*[\s\S]*?(?={re.escape(CHART_MARKER)}|<p class=\"sub-hdr\">|<div class=\"section\">)",
            block,
            html,
            count=1,
        )
    else:
        m = re.search(r'(<div class="page">\s*)', html)
        if m:
            html = html[: m.end()] + block + html[m.end() :]
        else:
            html = html.replace("</body>", block + "</body>")
    html_path.write_text(html, encoding="utf-8")


def _ensure_theme_css(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    if "--navy:" in html:
        return
    from evals.reports.utils.theme import REPORT_THEME_CSS

    if "</style>" in html:
        html = html.replace("</style>", REPORT_THEME_CSS + "</style>", 1)
        html_path.write_text(html, encoding="utf-8")


def inject_charts_section(
    html_path: Path,
    section_html: str,
    *,
    at_top: bool = True,
) -> None:
    """Insert or replace chart block — default ``at_top`` (before first stats table section)."""
    html = html_path.read_text(encoding="utf-8")
    block = f"{CHART_MARKER}\n{section_html}\n"
    _ensure_theme_css(html_path)
    html = html_path.read_text(encoding="utf-8")
    if CHART_MARKER in html:
        html = re.sub(
            rf"{re.escape(CHART_MARKER)}\s*[\s\S]*?(?=<p class=\"sub-hdr\">|<div class=\"section\">|</body>)",
            block,
            html,
            count=1,
        )
    elif at_top:
        anchor = re.search(
            r'<p class="sub-hdr">Detail tables|<div class="section">',
            html,
        )
        if anchor:
            html = html[: anchor.start()] + block + html[anchor.start() :]
        elif NARRATIVE_MARKER in html:
            html = html.replace(NARRATIVE_MARKER, f"{NARRATIVE_MARKER}\n{block}", 1)
        else:
            html = html.replace("</body>", block + "</body>")
    else:
        html = html.replace("</body>", block + "</body>")
    html_path.write_text(html, encoding="utf-8")
