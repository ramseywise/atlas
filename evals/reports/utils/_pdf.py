"""HTML → PDF via WeasyPrint (used by stats/suite renderers)."""

from __future__ import annotations

from pathlib import Path


def html_to_pdf(html_path: Path, output_path: Path | None = None) -> Path:
    try:
        from weasyprint import HTML
    except ImportError as e:
        raise ImportError("weasyprint not installed — run: uv add weasyprint") from e

    pdf_path = output_path or html_path.with_suffix(".pdf")
    HTML(filename=str(html_path.resolve())).write_pdf(str(pdf_path))
    return pdf_path
